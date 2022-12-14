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
from sklearn.metrics import precision_recall_fscore_support as pr
import transformers

from transformers import ViTConfig, ViTFeatureExtractor, ViTModel
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers import BartConfig, BartTokenizerFast, T5Config, T5TokenizerFast

from mydatasets import MustardCollator, MustardDataset
from modules import BartForConditionalGeneration_3P as BartForConditionalGeneration_P
from transformers import TrainingArguments, Trainer
from torch.utils.data import Subset

from transformers import HfArgumentParser, set_seed
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


def get_model(backbone, prompt_len, prompt_mask, pool, use_bce_loss):
    def set_config_attr(config):
        setattr(config, "prompt_len", prompt_len)
        setattr(config, "prompt_mask", prompt_mask)
        setattr(config, "pool", pool)
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
    v_feature_extractor = ViTFeatureExtractor.from_pretrained(vit_path)
    wav2vec_path = "facebook/wav2vec2-base-960h"
    wav2vec_config = Wav2Vec2Config.from_pretrained(wav2vec_path)
    a_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
    model.model.encoder.set_vit(vit_path, vit_config, wav2vec_path, wav2vec_config)
    # re-init prompts by existing token embeddings
    ids = random.sample(range(model.model.shared.weight.shape[0]), model.model.encoder.prompt_embedding.weight.shape[0])
    model.model.encoder.prompt_embedding.weight.data = model.model.shared.weight.data[ids]
    return model, tokenizer, v_feature_extractor, a_feature_extractor


def freeze(model, tune="ve"):
    """freeze model params"""
    if tune != "all":
        for param in model.model.parameters():
            param.requires_grad = False
    if tune == "ve":
        for param in model.model.encoder.visual_embedding.parameters():
            param.requires_grad = True
    elif tune == "ae":
        for param in model.model.encoder.audio_embedding.parameters():
            param.requires_grad = True
    elif tune == "e":
        for param in model.model.encoder.audio_embedding.parameters():
            param.requires_grad = True
        for param in model.model.encoder.visual_embedding.parameters():
            param.requires_grad = True
    elif tune == "prompt" and hasattr(model.model.encoder, "prompt_embedding"):
        for param in model.model.encoder.prompt_embedding.parameters():
            param.requires_grad = True
    # layernorm for vis and prompt
    for param in model.model.encoder.layernorm_vis.parameters():
        param.requires_grad = True
    for param in model.model.encoder.layernorm_aud.parameters():
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
    # pool video frames
    pool: bool = field(default=True,)
    # tune all, ve, ae, e, prompt
    tune: str = field(default="e")

    # keep these just for now
    use_bce_loss: bool = field(default=True,)
    blacked: bool = field(default=False,)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(default="",)
    train_split: Optional[str] = field(default="mustard_speaker_d_train",)
    validation_split: Optional[str] = field(default="mustard_speaker_d_test",)
    test_split: Optional[str] = field(default="mustard_speaker_d_test",)
    max_train_samples: Optional[int] = field(default=None,)
    max_eval_samples: Optional[int] = field(default=10,)
    max_predict_samples: Optional[int] = field(default=None,)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
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
    model, tokenizer, v_feature_extractor, a_feature_extractor = get_model(model_args.backbone,
                                                    model_args.prompt_len,
                                                    model_args.prompt_mask,
                                                    model_args.pool,
                                                    model_args.use_bce_loss,)
    model = freeze(model, model_args.tune)

    # Preprocessing the datasets.
    # Training preprocessing
    if training_args.do_train:
        train_dataset = MustardDataset(arrow_path=data_args.data_dir+f"{data_args.train_split}.arrow")
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            indices, indices_neg = [], []
            for index in range(len(train_dataset)):
                if str(train_dataset[index]["labels"]) == "True" and len(indices) < data_args.max_train_samples//2:
                    indices.append(index)
                if str(train_dataset[index]["labels"]) == "False" and len(indices_neg) < data_args.max_train_samples//2:
                    indices_neg.append(index)
            indices.extend(indices_neg)
            # indices = random.sample(range(len(train_dataset)), data_args.max_train_samples)
            train_dataset = Subset(train_dataset, indices)

    # Validation preprocessing
    if training_args.do_eval:
        eval_dataset = MustardDataset(arrow_path=data_args.data_dir+f"{data_args.validation_split}.arrow")
        if data_args.max_eval_samples is not None:
            eval_dataset = Subset(eval_dataset, list(range(data_args.max_eval_samples)))

    # Prediction preprocessing
    if training_args.do_predict:
        predict_dataset = MustardDataset(arrow_path=data_args.data_dir+f"{data_args.test_split}.arrow")
        if data_args.max_predict_samples is not None:
            predict_dataset = Subset(predict_dataset, list(range(data_args.max_predict_samples)))

    # Data collator
    data_collator = MustardCollator(a_feature_extractor, v_feature_extractor, tokenizer, model_args.blacked)
    label_idx = 1

    def compute_metrics(eval_preds):
        """for T/F"""
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = preds[:, label_idx, [36948, 46659]].argmax(axis=1)
        decoded_preds = [1 if i == 0 else 0 for i in preds]
        decoded_labels = [1 if label[label_idx] == 36948 else 0 for label in labels]
        logger.info(str((decoded_preds[:10], decoded_labels[:10])))
        Precis, Recall, Fscore, _ = pr(decoded_labels, decoded_preds, average='binary')
        # acc = np.array([decoded_preds[i] == decoded_labels[i] for i in range(len(decoded_preds))]).mean()
        f_pre_rate = sum([decoded_preds[i] == 0 for i in range(len(decoded_preds))]) / len(decoded_labels)
        result = {"Precis": Precis, "Recall": Recall, "Fscore": Fscore,
                  "f_pre_rate": round(f_pre_rate, 4) * 100}
            # "acc": round(acc, 4) * 100,

        return result

    # Initialize our Trainer
    trainer = Trainer(
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
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # bce case
        metrics = trainer.evaluate(metric_key_prefix="eval")
        # metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Evaluate ***")
        # bce case
        # metrics = trainer.evaluate(metric_key_prefix="eval")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_eval_samples = data_args.max_eval_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        metrics["predict_samples"] = min(max_eval_samples, len(predict_dataset))

        trainer.log_metrics("pre", metrics)
        trainer.save_metrics("pre", metrics)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
