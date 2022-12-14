import math
import random
from dataclasses import dataclass

from transformers.models.bart.modeling_bart import (
    BartLearnedPositionalEmbedding,
    BartEncoderLayer, BartEncoder, BartDecoder,
    BartPretrainedModel, BartModel, BartForConditionalGeneration,
    BartConfig,
    ACT2FN,
    shift_tokens_right, _make_causal_mask, _expand_mask
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, Module, Linear

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer
from transformers import ViTConfig, ViTFeatureExtractor, ViTModel, Wav2Vec2FeatureExtractor, Wav2Vec2Config, Wav2Vec2Model

logger = logging.get_logger(__name__)


class SeqPooler(Module):
    """pool seq len"""
    def __init__(self, in_seq_len, out_seq_len):
        super().__init__()
        self.if_seq_proj = in_seq_len != out_seq_len
        if self.if_seq_proj:
            self.seq_proj = Linear(in_seq_len, out_seq_len)

    def forward(self, x):
        if self.if_seq_proj:
            x = torch.einsum('...ij->...ji', [x])   # [8,197,1024] -> [8,1024,197]
            x = self.seq_proj(x)   # [8,1024,197] -> [8,1024,2]
            x = torch.einsum('...ij->...ji', [x])   # [8,1024,2] -> [8,2,1024]
        return x


def _expand_mask_P(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None,
                   prompt_mask_type: Optional[int] = None, prompt_mask: Optional[torch.Tensor] = None):
    """
    Context cannot see prompt.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    expanded_mask_P = expanded_mask.clone()
    if prompt_mask is not None and prompt_mask.max() > 0:
        prompt_mask = prompt_mask.to(dtype)
        if prompt_mask_type == 1:
            # prompt cannot see context
            for i in range(len(prompt_mask)):
                expanded_mask_P[i][:, (prompt_mask[i] == 1).nonzero().squeeze(1).tolist()] = prompt_mask[i]
        elif prompt_mask_type == 2:
            # context cannot see prompt
            for i in range(len(prompt_mask)):
                expanded_mask_P[i][0].T[(prompt_mask[i] == 1).nonzero().squeeze(1).tolist()] = prompt_mask[i]
    inverted_mask = 1.0 - expanded_mask_P
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class BartEncoder_P(BartEncoder):
    """
    BartEncoder + ViT + Wav2Vec + PromptEmbedding
    """
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        self.config = config

        # ViT and prompt here
        vit_config = ViTConfig()
        wav_config = Wav2Vec2Config()
        self.visual_embedding = ViTModel(vit_config)
        self.audio_embedding = Wav2Vec2Model(wav_config)
        self.prompt_embedding = nn.Embedding(50, config.d_model)
        # layernorms for vis and prompt
        self.layernorm_vis = nn.LayerNorm(config.d_model)
        self.layernorm_aud = nn.LayerNorm(config.d_model)
        self.layernorm_prompt = nn.LayerNorm(config.d_model)
        self.init_weights()
        self.layernorm_aud.weight.data = self.layernorm_embedding.weight.data
        self.layernorm_aud.bias.data = self.layernorm_embedding.bias.data
        self.layernorm_vis.weight.data = self.layernorm_embedding.weight.data
        self.layernorm_vis.bias.data = self.layernorm_embedding.bias.data
        self.layernorm_prompt.weight.data = self.layernorm_embedding.weight.data
        self.layernorm_prompt.bias.data = self.layernorm_embedding.bias.data

    def set_vit(self, vit_path, vit_config, wav_path, wav_config):
        self.visual_embedding = self.visual_embedding.from_pretrained(vit_path, config=vit_config)
        self.audio_embedding = self.audio_embedding.from_pretrained(wav_path, config=wav_config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,

            vis_feas=None,
            aud_feas=None,

            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            B, L = input_ids.size()
            input_ids = input_ids.view(-1, L)
        elif inputs_embeds is not None:
            B, L = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        """Text"""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype,
                                                                       device=inputs_embeds.device)
        inputs_embeds = self.layernorm_embedding(inputs_embeds + self.embed_positions((B, L)))

        """Video"""
        if vis_feas is not None:
            vis_embeds = torch.Tensor(())
            for vis_fea in vis_feas:
                vis_embeds = torch.cat((vis_embeds.to(vis_fea.device),
                                        self.visual_embedding(vis_fea)[1].unsqueeze(1)), dim=1)
            vis_embeds = self.layernorm_vis(vis_embeds + self.embed_positions(vis_embeds.size()[:-1]))
            if self.config.pool:
                vis_embeds = vis_embeds.mean(dim=1).unsqueeze(1)
            # print("v",vis_embeds.shape)
        else:
            vis_embeds = None

        """Audio"""
        if aud_feas is not None:
            aud_embeds = self.audio_embedding(aud_feas)[0]
            # aud_nonzro = (aud_embeds == 0).sum(dim=1)
            # if self.config.pool:
            # aud_embeds = (aud_embeds.sum(dim=1)/aud_nonzro).unsqueeze(1)
            aud_embeds = (self.layernorm_aud(aud_embeds + self.embed_positions(aud_embeds.size()[:-1]))).mean(dim=1).unsqueeze(1)
            # aud_embeds = self.layernorm_aud(aud_embeds)

        else:
            aud_embeds = None

        # concat V, A, T
        dtype = attention_mask.dtype
        device = attention_mask.device
        if aud_embeds is not None:
            aud_size = aud_embeds.size()[:-1]
            inputs_embeds = torch.cat((aud_embeds, inputs_embeds), dim=1)
            attention_mask = torch.cat((torch.ones(aud_size, dtype=dtype, device=device),
                                        attention_mask), dim=1)
        if vis_embeds is not None:
            vis_size = vis_embeds.size()[:-1]
            inputs_embeds = torch.cat((vis_embeds, inputs_embeds), dim=1)
            attention_mask = torch.cat((torch.ones(vis_size, dtype=dtype, device=device),
                                        attention_mask), dim=1)

        # concat with prompt (NOTE: when prompt_len=0 this still works
        prompt_embeds = self.prompt_embedding(torch.tensor([range(self.config.prompt_len)]*B,
                                                           dtype=torch.long, device=inputs_embeds.device))
        prompt_embeds = self.layernorm_prompt(prompt_embeds + self.embed_positions((B, self.config.prompt_len)))
        prompt_mask = torch.ones((B, self.config.prompt_len), dtype=dtype, device=device)

        attention_mask = torch.cat((prompt_mask, attention_mask), dim=1)
        prompt_mask = torch.cat((prompt_mask,
                                 torch.zeros(inputs_embeds.size()[:-1], dtype=dtype, device=device)), dim=1)
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)
        # expand attention mask [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # print(attention_mask.shape)
        attention_mask = _expand_mask_P(mask=attention_mask, dtype=inputs_embeds.dtype,
                                        prompt_mask_type=self.config.prompt_mask, prompt_mask=prompt_mask)
        hidden_states = inputs_embeds
        # hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class BartDecoder_P(BartDecoder):

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BartModel_P(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = BartEncoder_P(config, self.shared)
        self.decoder = BartDecoder_P(config, self.shared)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,

            vis_feas=None,
            aud_feas=None,

            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

            **kwargs,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vis_feas=vis_feas,
                aud_feas=aud_feas,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # add V, A, prompt to encoder_attention_mask
        # print(attention_mask.shape)
        add_mask_len = 0
        if vis_feas is not None:
            add_mask_len += 1 if self.config.pool else 20
        if aud_feas is not None:
            add_mask_len += 1
        attention_mask = torch.cat((torch.ones((attention_mask.size()[0], add_mask_len),
                                               dtype=attention_mask.dtype, device=attention_mask.device),
                                    attention_mask), dim=1)

        prompt_attention_mask = torch.ones((attention_mask.size()[0], self.config.prompt_len),
                                           dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat((prompt_attention_mask, attention_mask), dim=1)
        # print(attention_mask.shape)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class BartForConditionalGeneration_P(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"encoder\.visual_embedding\.*",
        r"encoder\.visual_projector\.*",
        r"encoder\.prompt_embedding\.*",
        r"encoder\.joint_projector\.*",
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
    ]

    def __init__(self, config: BartConfig):
        super(BartForConditionalGeneration, self).__init__(config)
        self.model = BartModel_P(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,

            vis_feas=None,
            aud_feas=None,

            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,

            vis_feas=vis_feas,
            aud_feas=aud_feas,

            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if self.config.use_bce_loss:
                label_idx = 1
                lm_logits_b = lm_logits[:, label_idx, [36948, 46659]]
                masked_lm_loss = loss_fct(lm_logits_b, (labels[:, label_idx] == 46659).to(torch.int64))
            else:
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: torch.LongTensor = None,
            encoder_outputs: ModelOutput = None,
            **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)
        if model_kwargs.get("vis_feas", None) is not None:
            for i in range(len(model_kwargs['vis_feas'])):
                model_kwargs['vis_feas'][i] = model_kwargs['vis_feas'][i].index_select(0, expanded_return_idx)
        if model_kwargs.get("aud_feas", None) is not None:
            model_kwargs['aud_feas'] = model_kwargs['aud_feas'].index_select(0, expanded_return_idx)
        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs