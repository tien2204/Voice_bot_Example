"""Fastformer attention definition.

Reference:
    Wu et al., "Fastformer: Additive Attention Can Be All You Need"
    https://arxiv.org/abs/2108.09084
    https://github.com/wuch15/Fastformer

"""

import numpy
import torch


class FastSelfAttention(torch.nn.Module):
    """Fast self-attention used in Fastformer."""

    def __init__(
        self,
        size,
        attention_heads,
        dropout_rate,
    ):
        super().__init__()
        if size % attention_heads != 0:
            raise ValueError(
                f"Hidden size ({size}) is not an integer multiple "
                f"of attention heads ({attention_heads})"
            )
        self.attention_head_size = size // attention_heads
        self.num_attention_heads = attention_heads

        self.query = torch.nn.Linear(size, size)
        self.query_att = torch.nn.Linear(size, attention_heads)
        self.key = torch.nn.Linear(size, size)
        self.key_att = torch.nn.Linear(size, attention_heads)
        self.transform = torch.nn.Linear(size, size)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        """Reshape and transpose to compute scores.

        Args:
            x: (batch, time, size = n_heads * attn_dim)

        Returns:
            (batch, n_heads, time, attn_dim)
        """

        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.reshape(*new_x_shape).transpose(1, 2)

    def forward(self, xs_pad, mask):
        """Forward method.

        Args:
            xs_pad: (batch, time, size = n_heads * attn_dim)
            mask: (batch, 1, time), nonpadding is 1, padding is 0

        Returns:
            torch.Tensor: (batch, time, size)
        """

        batch_size, seq_len, _ = xs_pad.shape
        mixed_query_layer = self.query(xs_pad)  
        mixed_key_layer = self.key(xs_pad)  

        if mask is not None:
            mask = mask.eq(0)  

        
        query_for_score = (
            self.query_att(mixed_query_layer).transpose(1, 2)
            / self.attention_head_size**0.5
        )
        if mask is not None:
            min_value = float(
                numpy.finfo(
                    torch.tensor(0, dtype=query_for_score.dtype).numpy().dtype
                ).min
            )
            query_for_score = query_for_score.masked_fill(mask, min_value)
            query_weight = torch.softmax(query_for_score, dim=-1).masked_fill(mask, 0.0)
        else:
            query_weight = torch.softmax(query_for_score, dim=-1)

        query_weight = query_weight.unsqueeze(2)  
        query_layer = self.transpose_for_scores(
            mixed_query_layer
        )  

        pooled_query = (
            torch.matmul(query_weight, query_layer)
            .transpose(1, 2)
            .reshape(-1, 1, self.num_attention_heads * self.attention_head_size)
        )  
        pooled_query = self.dropout(pooled_query)
        pooled_query_repeat = pooled_query.repeat(1, seq_len, 1)  

        mixed_query_key_layer = (
            mixed_key_layer * pooled_query_repeat
        )  

        
        query_key_score = (
            self.key_att(mixed_query_key_layer) / self.attention_head_size**0.5
        ).transpose(1, 2)
        if mask is not None:
            min_value = float(
                numpy.finfo(
                    torch.tensor(0, dtype=query_key_score.dtype).numpy().dtype
                ).min
            )
            query_key_score = query_key_score.masked_fill(mask, min_value)
            query_key_weight = torch.softmax(query_key_score, dim=-1).masked_fill(
                mask, 0.0
            )
        else:
            query_key_weight = torch.softmax(query_key_score, dim=-1)

        query_key_weight = query_key_weight.unsqueeze(2)  
        key_layer = self.transpose_for_scores(
            mixed_query_key_layer
        )  
        pooled_key = torch.matmul(
            query_key_weight, key_layer
        )  
        pooled_key = self.dropout(pooled_key)

        
        weighted_value = (pooled_key * query_layer).transpose(
            1, 2
        )  
        weighted_value = weighted_value.reshape(
            weighted_value.shape[:-2]
            + (self.num_attention_heads * self.attention_head_size,)
        )  
        weighted_value = (
            self.dropout(self.transform(weighted_value)) + mixed_query_layer
        )

        return weighted_value
