"""
Segmentaion Part 
Modified from DETR (https://github.com/facebookresearch/detr)
"""
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math 
from .attention_layer import _get_activation_fn

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.multihead_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, text,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     text_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     text_pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
                                   
        tgt_text = self.multihead_attn_text(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(text, text_pos),
                                   value=text,
                                   key_padding_mask=text_key_padding_mask)[0]
        tgt = tgt + tgt2 + tgt_text
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory, text,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    text_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    text_pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
                    
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        tgt_text = self.multihead_attn_text(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(text, text_pos),
                                   value=text,
                                   key_padding_mask=text_key_padding_mask)[0]

        tgt = tgt + tgt2 + tgt_text
        
        return tgt

    def forward(self, tgt, memory, text,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                text_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                text_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, text, memory_mask,
                                    memory_key_padding_mask, text_key_padding_mask, pos, text_pos, query_pos)
        return self.forward_post(tgt, memory, text, memory_mask,
                                    memory_key_padding_mask, text_key_padding_mask, pos, text_pos, query_pos)

# Bi-directional Attention Transfer
class BAT(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn_visual_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_text_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        tgt_lang = self.multihead_attn_text_1(
            query=self.with_pos_embed(memory, pos),
            key=self.with_pos_embed(tgt, query_pos),
            value=tgt
        )[0] * memory

        tgt_vis = self.multihead_attn_visual_1(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(tgt_lang, pos),
            value=tgt_lang, attn_mask=None,
            key_padding_mask=memory_key_padding_mask
        )[0] * tgt 

        return tgt_vis, tgt_lang
