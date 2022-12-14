#!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import random
import string
import transformers

from transformers import ViTConfig, ViTFeatureExtractor, ViTModel
from transformers import BartConfig, BartTokenizerFast, T5Config, T5TokenizerFast
from transformers import Seq2SeqTrainingArguments

from mydatasets import VQAv2Dataset, VQACollator, VQAEvaluator, normalize_word
from modules import BartForConditionalGeneration_P
from trainer import MySeq2SeqTrainer
from torch.utils.data import Subset

from transformers import HfArgumentParser, set_seed
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


def get_model(backbone, prompt_len, prompt_mask, prompt_pos, vis_proj, use_bce_loss):
    def set_config_attr(config):
        setattr(config, "prompt_len", prompt_len)
        setattr(config, "prompt_mask", prompt_mask)
        setattr(config, "prompt_pos", prompt_pos)
        setattr(config, "vis_proj", vis_proj)
        setattr(config, "use_bce_loss", use_bce_loss)
        setattr(config, "dropout", 0)   # turn off lm dropout
        return config
    # model or checkpoint
    tokenizer = BartTokenizerFast.from_pretrained(backbone)
    model_config = BartConfig.from_pretrained(backbone)
    model_config = set_config_attr(model_config)
    model = BartForConditionalGeneration_P.from_pretrained(backbone, config=model_config)
    # vit
    vit_path = 'google/vit-base-patch16-224-in21k'
    vit_config = ViTConfig.from_pretrained(vit_path)
    feature_extractor = ViTFeatureExtractor.from_pretrained(vit_path)
    model.model.encoder.set_vit(vit_path, vit_config)
    # re-init prompts by existing token embeddings
    ids = random.sample(range(model.model.shared.weight.shape[0]), model.model.encoder.prompt_embedding.weight.shape[0])
    model.model.encoder.prompt_embedding.weight.data = model.model.shared.weight.data[ids]
    return model, tokenizer, feature_extractor


def freeze(model, tune="ve"):
    """freeze model params"""
    if tune != "all":
        for param in model.model.parameters():
            param.requires_grad = False
    if tune == "ve":
        for param in model.model.encoder.visual_embedding.parameters():
            param.requires_grad = True
    elif tune == "vp" and hasattr(model.model.encoder, "visual_projector"):
        for param in model.model.encoder.visual_projector.parameters():
            param.requires_grad = True
    elif tune == "prompt" and hasattr(model.model.encoder, "prompt_embedding"):
        for param in model.model.encoder.prompt_embedding.parameters():
            param.requires_grad = True
    elif tune == "jp" and hasattr(model.model.encoder, "joint_projector"):
        for param in model.model.encoder.joint_projector.parameters():
            param.requires_grad = True
    # layernorm for vis and prompt
    for param in model.model.encoder.layernorm_vis.parameters():
        param.requires_grad = True
    for param in model.model.encoder.layernorm_prompt.parameters():
        param.requires_grad = True
    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    backbone: str = field(default='facebook/bart-base',)  # also checkpoints
    # prompt_len 0: no prompt
    prompt_len: int = field(default=0,)
    # prompt_mask 0: default attention mask; 1: prompt cant see context; 2: context cant see prompt
    prompt_mask: int = field(default=0,)
    # prompt position use for list.insert()
    prompt_pos: int = field(default=0,)
    # vis_proj 0: cls; 1: seq; 2: proj cls; 3: joint proj; -1: None
    vis_proj: int = field(default=0,)
    # tune all, ve, prompt, vp, jp
    tune: str = field(default="ve")

    # keep these just for now
    use_bce_loss: bool = field(default=False,)
    blacked: bool = field(default=False,)
    # verbalizer "tf" "yn" "number"
    verbalizer: str = field(default=None,)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(default="",)
    # Balanced_Binary_Abstract_Scenes Balanced_Real_Images
    train_split: Optional[str] = field(default="train",)
    validation_split: Optional[str] = field(default="val",)
    test_split: Optional[str] = field(default="val",)
    max_train_samples: Optional[int] = field(default=None,)
    max_eval_samples: Optional[int] = field(default=None,)
    max_predict_samples: Optional[int] = field(default=None,)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets:
    if data_args.data_dir is None:
        raise ValueError("Please enter dataset dir.")

    # Load pretrained model and tokenizer
    model, tokenizer, feature_extractor = get_model(model_args.backbone,
                                                    model_args.prompt_len,
                                                    model_args.prompt_mask,
                                                    model_args.prompt_pos,
                                                    model_args.vis_proj,
                                                    model_args.use_bce_loss,)
    model = freeze(model, model_args.tune)

    # verbalizer
    # a stupid way: set unrelated tokens' lm_head weights to a very negative number then they would be 0 in softmax
    ans = {"number": ["none", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
           "tf": ["true", "false"], "yn": ["yes", "no"]}
    if model_args.verbalizer is not None:
        a_id = tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens+ans[model_args.verbalizer])
        model.lm_head.weight.data[np.setdiff1d(range(len(tokenizer)), a_id)] = -1e+4

    # Preprocessing the datasets.
    # Training preprocessing
    if training_args.do_train:
        train_dataset = VQAv2Dataset(data_dir=data_args.data_dir, split=data_args.train_split)
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            indices = random.sample(range(len(train_dataset)), data_args.max_train_samples)
            train_dataset = Subset(train_dataset, indices)

    # Validation preprocessing
    if training_args.do_eval:
        eval_dataset = VQAv2Dataset(data_dir=data_args.data_dir, split=data_args.validation_split)
        if data_args.max_eval_samples is not None:
            eval_dataset = Subset(eval_dataset, list(range(data_args.max_eval_samples)))

    # Prediction preprocessing
    if training_args.do_predict:
        predict_dataset = VQAv2Dataset(data_dir=data_args.data_dir, split=data_args.test_split)
        if data_args.max_predict_samples is not None:
            predict_dataset = Subset(predict_dataset, list(range(data_args.max_predict_samples)))

    # Data collator
    data_collator = VQACollator(feature_extractor, tokenizer, model_args.blacked)

    def compute_metrics(eval_preds):
        """for generation case"""
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        print(preds.shape)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_preds = [normalize_word(p) for p in decoded_preds]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = [normalize_word(p) for p in decoded_labels]

        print(decoded_preds[:10], decoded_labels[:10])
        acc = np.array([decoded_preds[i] == decoded_labels[i] for i in range(len(decoded_preds))]).mean()
        result = {"acc": round(acc, 4) * 100}
        return result

    # Initialize our Trainer
    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = training_args.generation_max_length
    num_beams = training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                logger.info(str(predictions[:10]))
                # eval on predict dataset
                evaluator = VQAEvaluator(predict_dataset, predictions)
                accuracy = evaluator.evaluate_raw()
                logger.info("Prediction Acc ---> {0}".format(accuracy))
                # dump results
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                evaluator.dump_result(output_prediction_file)

                results = accuracy
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
