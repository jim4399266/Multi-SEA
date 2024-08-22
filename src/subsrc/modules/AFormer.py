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

class HSwish(nn.Module):
    def __init__(self, a_max: float=1.0, a_min: float=0.0, add: float=3.0, divide: float=6.0) -> None:
        super().__init__()
        self.a_max = a_max
        self.a_min = a_min
        self.add = add
        self.divide = divide

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.clip(input + self.add, max=self.a_max, min=self.a_min) / self.divide


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
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class AgentAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super(AgentAttention, self).__init__()
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
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)

        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "agent_position_embedding_type", "absolute")
        if self.position_embedding_type == "agent_relative_key" or self.position_embedding_type == "agent_relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False

        self.scale = self.attention_head_size ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.agent_num = 40
        self.dwc = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            past_key_value=None,
            output_attentions=False,
    ):
        '''
        :param hidden_states: [bs, seq_len, hidden_size]
        :param attention_mask:  [bs, 1, 1, seq_len]
        :param output_attentions:
        '''

        ## query, key, value : [bs, heads, seq_len, head_dim]
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        is_cross_attention = encoder_hidden_states is not None
        bs, seq_len, dim = hidden_states.size()
        bs_kv, seq_len_kv, dim_kv = encoder_hidden_states.size() if is_cross_attention else hidden_states.size()
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            agent_attention_mask, q_attention_mask = encoder_attention_mask, attention_mask.transpose(-1, -2)

        elif past_key_value is not None and use_cache:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            agent_attention_mask, q_attention_mask = attention_mask, attention_mask.transpose(-1, -2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            agent_attention_mask, q_attention_mask = attention_mask, attention_mask.transpose(-1, -2)

        ## agent_tokens : [bs, heads, agent_num, head_dim]
        q = query_layer.transpose(1, 2).reshape(bs, seq_len, dim)
        agent_tokens = F.interpolate(q.unsqueeze(0), size=(self.agent_num, dim), mode='bilinear', align_corners=False).squeeze(0)
        agent_tokens = self.transpose_for_scores(agent_tokens)

        ## Step 1, Agent Aggregation.  X = A @ K^T  @ V :
        # [bs, head_n, agent_n, head_dim]  @ [bs, head_n, head_dim, seq_len] @ [bs, head_n, seq_len, head_dim]  ----->
        # [bs, head_n, agent_n, head_dim]

        if self.position_embedding_type == "agent_relative_key" or self.position_embedding_type == "agent_relative_key_query":
            # seq_length = hidden_states.size()[1]
            position_agent_l = torch.arange(self.agent_num, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_agent_r = torch.arange(seq_len_kv, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_agent_l - position_agent_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=hidden_states.dtype)  # fp16 compatibility
            if self.position_embedding_type == "agent_relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", agent_tokens, positional_embedding)
                position_bias = relative_position_scores
            elif self.position_embedding_type == "agent_relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", agent_tokens, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                position_bias = relative_position_scores_query + relative_position_scores_key
        else:
            position_bias = 0.
        agent_attn_score = (torch.matmul(agent_tokens * self.scale, key_layer.transpose(-2, -1))) + position_bias
        if agent_attention_mask is not None:
            # attention_mask : [bs, head_n, agent_n, seq_len]
            agent_attn_score = agent_attn_score + agent_attention_mask
        agent_attn_probs = nn.Softmax(dim=-1)(agent_attn_score)
        agent_attn_probs_dropped = self.dropout(agent_attn_probs)
        agent_v = torch.matmul(agent_attn_probs_dropped, value_layer)

        ## Step 2, Agent Broadcast.  Y = Q @ A^T :
        # [bs, head_n, seq_len, head_dim] @ [bs, head_n, head_dim, agent_n] ----> [bs, head_n, seq_len, agent_n]
        if self.position_embedding_type == "agent_relative_key" or self.position_embedding_type == "agent_relative_key_query":
            # seq_length = hidden_states.size()[1]
            position_pos_l = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_pos_r = torch.arange(self.agent_num, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_pos_l - position_pos_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=hidden_states.dtype)  # fp16 compatibility
            if self.position_embedding_type == "agent_relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                agent_bias = relative_position_scores
            elif self.position_embedding_type == "agent_relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", agent_tokens, positional_embedding)
                agent_bias = relative_position_scores_query + relative_position_scores_key
        else:
            agent_bias = 0.

        q_attn_score = (torch.matmul(query_layer * self.scale, agent_tokens.transpose(-2, -1))) + agent_bias
        if q_attention_mask is not None:
            # attention_mask : [bs, head_n, seq_len, agent_n]
            q_attn_score = q_attn_score + q_attention_mask
        q_attn_probs = nn.Softmax(dim=-1)(q_attn_score)
        q_attn_probs_dropped = self.dropout(q_attn_probs)

        ## Step 3, Generalized Linear Attention.  Y @ X:
        # [bs, head_n, seq_len, agent_n] @ [bs, head_n, agent_n, head_dim] ----> [bs, head_n, seq_len, head_dim]
        x = torch.matmul(q_attn_probs_dropped, agent_v).transpose(1, 2).reshape(bs, seq_len, dim)

        outputs = (x, agent_attn_probs, q_attn_probs) if output_attentions else (x,)
        outputs = outputs + (past_key_value,)
        return outputs


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
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
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
            use_cache=False,
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
        elif past_key_value is not None and use_cache:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)


        past_key_value = (key_layer, value_layer) if use_cache else None

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

class AFormerAttentionOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_norm = RMSNorm(config.hidden_size)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.attention_norm(hidden_states + input_tensor)
        return hidden_states
class AFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = AFormerSelfAttention(config, is_cross_attention)
        self.output = AFormerAttentionOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            past_key_value=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            past_key_value,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class AFormerAgentAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.attn = AgentAttention(config, is_cross_attention)
        self.output = AFormerAttentionOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            past_key_value=None,
            output_attentions=False,
    ):
        self_outputs = self.attn(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            past_key_value,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
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

    def forward(self, input_tensor):
        hidden_states = self.swish(self.dense1(input_tensor))
        hidden_states = hidden_states * self.dense3(input_tensor)
        return self.ffn_norm(input_tensor + self.dropout(self.dense2(hidden_states)))

class AFormerGateFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense3 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense4 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.swish = Swish(config.beta)
        # self.swish = HSwish()

    def forward(self, input_tensor):
        hidden_states = self.swish(self.dense1(input_tensor))
        hidden_states = hidden_states * self.dropout(self.dense2(input_tensor))

        output_tensor = self.swish(self.dense3(hidden_states))
        output_tensor = output_tensor * self.dropout(self.dense4(hidden_states))

        return self.ffn_norm(input_tensor + output_tensor)


class AFormerCrossGateFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense3 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense4 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ffn_norm = RMSNorm(config.hidden_size)
        # self.swish = Swish(config.beta)
        self.swish = HSwish()

    def forward(self, input_tensor):
        hidden_states1 = self.swish(self.dense1(input_tensor))
        hidden_states2 = self.dropout(self.dense2(input_tensor))
        hidden_states3 = hidden_states1 * hidden_states2

        output_tensor1 = self.dropout(self.dense3(hidden_states3))
        output_tensor2 = self.swish(self.dense4(hidden_states2))

        output_tensor = output_tensor1 * output_tensor2

        return self.ffn_norm(input_tensor + output_tensor)

class AFormerLayerExpert(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = AFormerAttention(config, is_cross_attention=False)
        self.cross_attention = AFormerAgentAttention(config, is_cross_attention=True)
        # self.ffn = AFormerFeedForward(config)
        # self.ffn = AFormerGateFeedForward(config)
        self.ffn = AFormerCrossGateFeedForward(config)
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


class AFormerWithAug(BertPreTrainedModel):
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

