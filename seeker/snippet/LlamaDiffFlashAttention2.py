#date: 2024-11-05T17:04:59Z
#url: https://api.github.com/gists/25e1a7f7ea8599aaaa0af7d5a9d04c9f
#owner: https://api.github.com/users/DSamuelHodge

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv
)


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class LlamaDiffFlashAttention2(LlamaAttention):
    """
    Llama Differential Flash Attention implementation using Flash Attention 2.
    This implements the differential attention mechanism while maintaining compatibility
    with the original flash attention optimizations.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        
        # Initialize lambda parameters
        self.layer_idx = layer_idx if layer_idx is not None else 0
        self.lambda_init = lambda_init_fn(self.layer_idx)
        
        
        # Initialize learnable lambda parameters
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        
        
        # RMSNorm for sub-layer normalization
        self.subln = nn.LayerNorm(2 * self.head_dim, eps=1e-6, elementwise_affine=True)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:


        bsz, q_len, _ = hidden_states.size()


        # Project input for queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)


        # Reshape and split for differential attention
        query_states = query_states.view(bsz, q_len, 2 * self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, 2 * self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, 2 * self.head_dim)


        # Apply rotary embeddings
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            position_embeddings = (cos, sin)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        # Reshape states for differential computation
        query_states = query_states.reshape(bsz, q_len, self.num_heads, 2, self.head_dim)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, 2, self.head_dim)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, 2, self.head_dim)


        # Split states for differential attention
        q1, q2 = query_states[:, :, :, 0], query_states[:, :, :, 1]
        k1, k2 = key_states[:, :, :, 0], key_states[:, :, :, 1]
        v1, v2 = value_states[:, :, :, 0], value_states[:, :, :, 1]


        # Handle key-value caching if needed
        if past_key_value is not None:
            k1, v1 = self._concat_with_cache(k1, v1, past_key_value[0], past_key_value[1], cache_position)
            k2, v2 = self._concat_with_cache(k2, v2, past_key_value[2], past_key_value[3], cache_position)
            past_key_value = (k1, v1, k2, v2)


        # Repeat keys and values for grouped query attention
        k1 = repeat_kv(k1, self.num_key_value_groups)
        k2 = repeat_kv(k2, self.num_key_value_groups)
        v1 = repeat_kv(v1, self.num_key_value_groups)
        v2 = repeat_kv(v2, self.num_key_value_groups)


        # Compute dropout rate (0 for inference)
        dropout_rate = self.attention_dropout if self.training else 0.0


        # Compute attention with flash attention
        attn11 = self._flash_attention_forward(q1, k1, v1, attention_mask, dropout_rate)
        attn12 = self._flash_attention_forward(q1, k1, v2, attention_mask, dropout_rate)
        attn1 = torch.cat([attn11, attn12], dim=-1)


        attn21 = self._flash_attention_forward(q2, k2, v1, attention_mask, dropout_rate)
        attn22 = self._flash_attention_forward(q2, k2, v2, attention_mask, dropout_rate)
        attn2 = torch.cat([attn21, attn22], dim=-1)


        # Compute lambda scaling factors
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).to(attn1.dtype)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).to(attn1.dtype)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init


        # Apply differential attention
        attn_output = attn1 - lambda_full * attn2


        # Apply sub-layer normalization and scaling
        attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)


        # Reshape and project output
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * 2 * self.head_dim)
        attn_output = self.o_proj(attn_output)


        if not use_cache:
            past_key_value = None


        return attn_output, None, past_key_value


    def _flash_attention_forward(
        self, 
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Helper function to compute flash attention for a single query-key-value set
        """
        # Import here to avoid circular imports
        from flash_attn import flash_attn_func


        # Flash attention expects shape (batch, seqlen, nheads, headdim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)


        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=dropout_p,
            causal=True,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
        )


        return attn_output.transpose(1, 2)


    def _concat_with_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor]


# Initialize the model
config = LlamaConfig(...)
attention_layer = LlamaDiffFlashAttention2(config, layer_idx=0)


# Forward pass
output, attention_weights, past_key_value = attention_layer(
    hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids
) = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function to concatenate current key/value states with cached states
        """
        if key_cache is not None and value_cache is not None:
            key_states = torch.cat([key_cache, key_states], dim=1)
            value_states = torch.cat([value_cache, value_states], dim=1)
        return key_states, value_states


# Initialize the model
config = LlamaConfig(...)
attention_layer = LlamaDiffFlashAttention2(config, layer_idx=0)


# Forward pass
output, attention_weights, past_key_value = attention_layer(
    hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids
)