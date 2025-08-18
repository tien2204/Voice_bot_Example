





"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn
from typing import Optional, Tuple

import torch.nn.functional as F
from src.models.src_step_audio.funasr_detach.models.transformer.utils.nets_utils import make_pad_mask
import src.models.src_step_audio.funasr_detach.models.lora.layers as lora


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (
            key (torch.Tensor): Key tensor (
            value (torch.Tensor): Value tensor (

        Returns:
            torch.Tensor: Transformed query tensor (
            torch.Tensor: Transformed key tensor (
            torch.Tensor: Transformed value tensor (

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)  

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (
            scores (torch.Tensor): Attention score (
            mask (torch.Tensor): Mask (

        Returns:
            torch.Tensor: Transformed value (
                weighted by the attention score (

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  
        else:
            self.attn = torch.softmax(scores, dim=-1)  

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  

        return self.linear_out(x)  

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (
            key (torch.Tensor): Key tensor (
            value (torch.Tensor): Value tensor (
            mask (torch.Tensor): Mask tensor (
                (

        Returns:
            torch.Tensor: Output tensor (

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        
        
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (
            key (torch.Tensor): Key tensor (
            value (torch.Tensor): Value tensor (
            pos_emb (torch.Tensor): Positional embedding tensor
                (
            mask (torch.Tensor): Mask tensor (
                (

        Returns:
            torch.Tensor: Output tensor (

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  

        
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        
        
        
        
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        
        
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  

        return self.forward_attention(v, scores, mask)


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, in_feat, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadSelfAttention, self).__init__()
        assert n_feat % n_head == 0
        
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, x):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (
            key (torch.Tensor): Key tensor (
            value (torch.Tensor): Value tensor (

        Returns:
            torch.Tensor: Transformed query tensor (
            torch.Tensor: Transformed key tensor (
            torch.Tensor: Transformed value tensor (

        """
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  

        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (
            scores (torch.Tensor): Attention score (
            mask (torch.Tensor): Mask (

        Returns:
            torch.Tensor: Transformed value (
                weighted by the attention score (

        """
        n_batch = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            mask = mask.unsqueeze(1).eq(0)  

            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  
        else:
            self.attn = torch.softmax(scores, dim=-1)  

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  

        return self.linear_out(x)  

    def forward(self, x, mask, mask_att_chunk_encoder=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (
            key (torch.Tensor): Key tensor (
            value (torch.Tensor): Value tensor (
            mask (torch.Tensor): Mask tensor (
                (

        Returns:
            torch.Tensor: Output tensor (

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs
