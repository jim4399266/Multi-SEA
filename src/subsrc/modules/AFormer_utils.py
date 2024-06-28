from typing import Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers import BertPreTrainedModel
from transformers.modeling_utils import apply_chunking_to_forward
from torch import Tensor, device, dtype, nn
import math

'''
AFormer组件
'''
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
        self.position_embedding_type = getattr(config, "agent_position_embedding_type", "absolute")
        if self.position_embedding_type == "agent_relative_key" or self.position_embedding_type == "agent_relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False

        self.scale = self.attention_head_size ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.agent_num = 40
        # self.sampler = F.interpolate(output_size=self.agent_num)
        # self.pooler = nn.AdaptiveAvgPool1d(output_size=self.agent_num)
        self.dwc = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)
        # self.window = 14

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
        # agent_tokens = self.transpose_for_scores(self.pooler(query_layer.transpose(1, 2).reshape(bs, seq_len, dim)))
        ## TODO agent_tokens 的采样方法
        q = query_layer.transpose(1, 2).reshape(bs, seq_len, dim)
        agent_tokens = F.interpolate(q.unsqueeze(0), size=(self.agent_num, dim), mode='bilinear', align_corners=False).squeeze()
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
        # TODO dwc的卷积方法
        # v = value_layer.transpose(1, 2).reshape(bs, -1, dim)
        # x = x + self.dwc(v.unsqueeze(1)).squeeze()
        # x = x + self.dwc(q.reshape(bs, -1, dim).unsqueeze(1)).squeeze()

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

class AFormerAugAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        # self.self = AFormerSelfAttention(config, is_cross_attention=False)
        self.a_attention = AgentAttention(config, is_cross_attention=True)
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
        aug_outputs = self.a_attention(
            hidden_states,
            attention_mask,)

        attention_output = self.output(self_outputs[0] + aug_outputs[0], hidden_states)
        ## TODO 调试
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_tensor):
        hidden_states = self.dense1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

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
        # hidden_states = self.swish(self.dense1(input_tensor)) * self.dense3(input_tensor)
        return self.ffn_norm(input_tensor + self.dropout(self.dense2(hidden_states)))