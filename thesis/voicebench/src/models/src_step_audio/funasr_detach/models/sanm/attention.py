





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


class MultiHeadedAttentionSANM(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        dropout_rate,
        kernel_size,
        sanm_shfit=0,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        
        self.d_k = n_feat // n_head
        self.h = n_head
        
        
        
        if lora_list is not None:
            if "o" in lora_list:
                self.linear_out = lora.Linear(
                    n_feat,
                    n_feat,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
            else:
                self.linear_out = nn.Linear(n_feat, n_feat)
            lora_qkv_list = ["q" in lora_list, "k" in lora_list, "v" in lora_list]
            if lora_qkv_list == [False, False, False]:
                self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
            else:
                self.linear_q_k_v = lora.MergedLinear(
                    in_feat,
                    n_feat * 3,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    enable_lora=lora_qkv_list,
                )
        else:
            self.linear_out = nn.Linear(n_feat, n_feat)
            self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

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

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
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
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs + fsmn_memory

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
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
        if chunk_size is not None and look_back > 0 or look_back == -1:
            if cache is not None:
                k_h_stride = k_h[:, :, : -(chunk_size[2]), :]
                v_h_stride = v_h[:, :, : -(chunk_size[2]), :]
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)

                cache["k"] = torch.cat((cache["k"], k_h_stride), dim=2)
                cache["v"] = torch.cat((cache["v"], v_h_stride), dim=2)
                if look_back != -1:
                    cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                    cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
            else:
                cache_tmp = {
                    "k": k_h[:, :, : -(chunk_size[2]), :],
                    "v": v_h[:, :, : -(chunk_size[2]), :],
                }
                cache = cache_tmp
        fsmn_memory = self.forward_fsmn(v, None)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, None)
        return att_outs + fsmn_memory, cache


class MultiHeadedAttentionSANMDecoder(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_feat, dropout_rate, kernel_size, sanm_shfit=0):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttentionSANMDecoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        
        
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.kernel_size = kernel_size

    def forward(self, inputs, mask, cache=None, mask_shfit_chunk=None):
        """
        :param x: (
        :param mask: Mask tensor (
        :return:
        """
        
        b, t, d = inputs.size()
        
        
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            
            if mask_shfit_chunk is not None:
                
                mask = mask * mask_shfit_chunk
            
            
            
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        b, d, t = x.size()
        if cache is None:
            

            x = self.pad_fn(x)
            if not self.training:
                cache = x
        else:
            
            
            
            
            x = torch.cat((cache[:, :, 1:], x), dim=2)
            x = x[:, :, -(self.kernel_size + t - 1) :]
            
            cache = x
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        
        if x.size(1) != inputs.size(1):
            inputs = inputs[:, -1, :]

        x = x + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x, cache


class MultiHeadedAttentionCrossAtt(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        encoder_output_size=None,
    ):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttentionCrossAtt, self).__init__()
        assert n_feat % n_head == 0
        
        self.d_k = n_feat // n_head
        self.h = n_head
        if lora_list is not None:
            if "q" in lora_list:
                self.linear_q = lora.Linear(
                    n_feat,
                    n_feat,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
            else:
                self.linear_q = nn.Linear(n_feat, n_feat)
            lora_kv_list = ["k" in lora_list, "v" in lora_list]
            if lora_kv_list == [False, False]:
                self.linear_k_v = nn.Linear(
                    n_feat if encoder_output_size is None else encoder_output_size,
                    n_feat * 2,
                )
            else:
                self.linear_k_v = lora.MergedLinear(
                    n_feat if encoder_output_size is None else encoder_output_size,
                    n_feat * 2,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    enable_lora=lora_kv_list,
                )
            if "o" in lora_list:
                self.linear_out = lora.Linear(
                    n_feat,
                    n_feat,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
            else:
                self.linear_out = nn.Linear(n_feat, n_feat)
        else:
            self.linear_q = nn.Linear(n_feat, n_feat)
            self.linear_k_v = nn.Linear(
                n_feat if encoder_output_size is None else encoder_output_size,
                n_feat * 2,
            )
            self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, x, memory):
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

        
        b = x.size(0)
        q = self.linear_q(x)
        q_h = torch.reshape(q, (b, -1, self.h, self.d_k)).transpose(
            1, 2
        )  

        k_v = self.linear_k_v(memory)
        k, v = torch.split(k_v, int(self.h * self.d_k), dim=-1)
        k_h = torch.reshape(k, (b, -1, self.h, self.d_k)).transpose(
            1, 2
        )  
        v_h = torch.reshape(v, (b, -1, self.h, self.d_k)).transpose(
            1, 2
        )  

        return q_h, k_h, v_h

    def forward_attention(self, value, scores, mask, ret_attn=False):
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
        if ret_attn:
            return self.linear_out(x), self.attn  
        return self.linear_out(x)  

    def forward(self, x, memory, memory_mask, ret_attn=False):
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
        q_h, k_h, v_h = self.forward_qkv(x, memory)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        return self.forward_attention(v_h, scores, memory_mask, ret_attn=ret_attn)

    def forward_chunk(self, x, memory, cache=None, chunk_size=None, look_back=0):
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
        q_h, k_h, v_h = self.forward_qkv(x, memory)
        if chunk_size is not None and look_back > 0:
            if cache is not None:
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)
                cache["k"] = k_h[:, :, -(look_back * chunk_size[1]) :, :]
                cache["v"] = v_h[:, :, -(look_back * chunk_size[1]) :, :]
            else:
                cache_tmp = {
                    "k": k_h[:, :, -(look_back * chunk_size[1]) :, :],
                    "v": v_h[:, :, -(look_back * chunk_size[1]) :, :],
                }
                cache = cache_tmp
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        return self.forward_attention(v_h, scores, None), cache


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
