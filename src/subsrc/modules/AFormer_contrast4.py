from typing import Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers import BertPreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_utils import apply_chunking_to_forward
from torch import Tensor, device, dtype, nn
import math
from .AFormer_utils import AFormerAttention, Intermediate, RMSNorm, Pooler

class AFormerLayerExpert(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = AFormerAttention(config, is_cross_attention=False)
        self.cross_attention = AFormerAttention(config, is_cross_attention=True)
        self.ffn = Intermediate(config)
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=False,
                past_kye_value=None,
                output_attentions=False,
                mode=None
                ):
        self_attn_past_key_value = past_kye_value[:2] if past_kye_value is not None else None
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            head_mask,
            use_cache=use_cache,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if mode == 't2i' or mode == 'i2t':
            assert encoder_hidden_states is not None, "encoder_hidden_states must be given for cross-attention layers"
            cross_attention_outputs = self.cross_attention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:-1]

        layer_output = apply_chunking_to_forward(
            self.ffn.forward, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs


class AFormerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_expert = AFormerLayerExpert(config)
        self.image_expert = AFormerLayerExpert(config)


    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=False,
                past_kye_value=None,
                output_attentions=False,
                mode=None
                ):
        assert mode in [None, 'text', 'image', 't2i', 'i2t']
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2

        if mode == 'text' or mode == 't2i':
            outputs = self.text_expert(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache=use_cache,
                past_kye_value=past_kye_value,
                output_attentions=output_attentions,
                mode=mode,
            )
        elif mode == 'image' or mode == 'i2t':
            outputs = self.image_expert(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache=use_cache,
                past_kye_value=past_kye_value,
                output_attentions=output_attentions,
                mode=mode,
            )
        else:
            raise NotImplementedError

        return outputs

class AFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AFormerLayer(config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                mode=None,
                ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        # -----------------------------------------------------------
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layers[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_values = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                past_key_values,
                output_attentions,
                mode=mode,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        # -----------------------------------------------------------
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class AFormerContrast4(BertPreTrainedModel):
    '''
    不使用Agent attention和SwiFFN
    '''
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = AFormerEncoder(config)
        self.pooler = Pooler(config.hidden_size)
        self.init_weights()

    def forward(self,
                inputs_embeds,
                attention_mask=None,
                head_mask=None,
                encoder_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                is_decoder=False,
                mode=None,
                ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif encoder_embeds is not None:
            input_shape = encoder_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = encoder_embeds.device
        else:
            raise ValueError("You have to specify either inputs_embeds or encoder_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=mode,
        )
        sequence_output = encoder_outputs[0]
        # pooled_output = F.normalize(self.pooler(sequence_output)) if self.pooler is not None else None
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

