from typing import Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers import BertPreTrainedModel
from transformers.modeling_utils import apply_chunking_to_forward
from torch import Tensor, device, dtype, nn
import math


class Swish(nn.Module):
    def __init__(self, beta: float=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, input: Tensor) -> Tensor:
        return input * (1 / (1 + torch.exp(-self.beta * input)))

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class GQA_Linear_new(nn.Linear):
    '''
        从dim=0维度进行共享，变换矩阵：[hidden_size // groups, all_head_size]
    '''
    def __init__(self, in_features: int, heads: int, per_head_size:int, attention_groups: int = 1, bias: bool = True,
                 device=None, dtype=None):
        super().__init__(in_features, heads * per_head_size, bias, device, dtype)
        # self.linear = nn.Linear(in_features, heads * per_head_size, bias, device, dtype)
        self.n_rep = attention_groups
        self.heads = heads
        self.per_head_size = per_head_size

    def forward(self, input: Tensor) -> Tensor:
        bs, slen, dim = input.shape
        span = dim // self.n_rep
        x = torch.stack([F.linear(input[:, :, i * span: (i+1) *span], self.weight, self.bias) for i in range(self.n_rep)], dim=0)
        x = torch.sum(x, dim=0)
        return x
class GQA_Linear(nn.Linear):
    '''
    从dim=1维度进行共享，变换矩阵：[hidden_size, all_head_size // groups]
    '''
    def __init__(self, in_features: int, heads: int, per_head_size:int, attention_groups: int = 1, bias: bool = True,
                 device=None, dtype=None):
        super().__init__(in_features, heads * per_head_size, bias, device, dtype)
        # self.linear = nn.Linear(in_features, heads * per_head_size, bias, device, dtype)
        self.n_rep = attention_groups
        self.heads = heads
        self.per_head_size = per_head_size

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep * head_dim)
        )
    def forward(self, input: Tensor) -> Tensor:
        bs, slen, dim = input.shape
        x = F.linear(input, self.weight, self.bias)
        x = x.view(bs, slen, self.heads, self.per_head_size)
        x = self.repeat_kv(x, self.n_rep)
        assert x.shape[-1] == dim
        return x

class AFormerSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            if not config.attention_groups:
                self.key = nn.Linear(config.encoder_width, self.all_head_size)
                self.value = nn.Linear(config.encoder_width, self.all_head_size)
            # else:
            #     ## Group-Query Attention
            #     kv_head = self.num_attention_heads // config.attention_groups
            #     self.key = GQA_Linear(config.encoder_width, kv_head, self.attention_head_size, config.attention_groups)
            #     self.value = GQA_Linear(config.encoder_width, kv_head, self.attention_head_size, config.attention_groups)

            else:
                ## Group-Query Attention 1
                in_features = config.encoder_width // config.attention_groups
                self.key = GQA_Linear_new(in_features, self.num_attention_heads, self.attention_head_size,
                                       config.attention_groups)
                self.value = GQA_Linear_new(in_features, self.num_attention_heads, self.attention_head_size,
                                         config.attention_groups)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

            # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


class AFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.self = AFormerSelfAttention(config, is_cross_attention)

        # self.output = BertSelfOutput(config)
        # self.pruned_heads = set()

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        hidden_states = self.attention_norm(hidden_states)
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self_outputs[0] + self.dropout(self.dense(self_outputs[0]))
        # attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class AFormerFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense3 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.swish = Swish(config.beta)

    def forward(self, x):
        x = self.ffn_norm(x)
        x = self.swish(self.dense1(x)) * self.dense3(x)
        return self.dropout(self.dense2(x))

class AFormerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len_dim = 1
        self.chunk_size_feed_forward = 0
        self.self_attention = AFormerAttention(config)
        self.t_cross_attention = AFormerAttention(config, is_cross_attention=True)
        self.i_cross_attention = AFormerAttention(config, is_cross_attention=True)
        self.ffn = AFormerFeedForward(config)
        # self.intermediate = BertIntermediate(config)
        # self.output = BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_kye_value=None,
                output_attentions=False,
                mode=None
                ):
        assert mode in [None, 'text', 'image', 't2i', 'i2t']
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_kye_value[:2] if past_kye_value is not None else None
        self_attention_outputs = self.self_attention(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    output_attentions=output_attentions,
                    past_key_value=self_attn_past_key_value,
                )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if mode == 't2i' or mode == 'i2t':
            assert encoder_hidden_states is not None, "encoder_hidden_states must be given for cross-attention layers"
            if mode == 't2i':
                cross_attention_outputs = self.t_cross_attention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
            else:
                cross_attention_outputs = self.i_cross_attention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
            attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:-1]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output, ) + outputs
        outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        # intermediate_output = self.intermediate(attention_output)
        # layer_output = self.output(intermediate_output, attention_output)
        layer_output = self.ffn(attention_output)
        return layer_output

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

        hidden_states = self.norm(hidden_states)

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


class AFormer(BertPreTrainedModel):
    '''
    A_Former作为桥梁连接两个模态的信息。每层包含 self-attention, cross-attention, ffn三个模块。
    AFormer首先接收 各自模态的 last_hidden_state[bs, len, dim]，遮住cross-attention，得到 feature
    利用 feature 进行对比学习得到 候选集，候选集通过 cross-attention 精排。
    '''
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = AFormerEncoder(config)
        self.pooler = Pooler(config)
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
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
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

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
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

