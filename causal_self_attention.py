import os
import math
import time
import inspect
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import dataclass
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
from collections import OrderedDict
from dilated_attention import MixedDilatedAttention
from differential_attention import DifferentialFlashAttention

torch.cuda.empty_cache()  # Clear the cache
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd).to(device)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd).to(device)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, kv_cache=None, use_cache=False, output_attentions=False):
        B, T, C = x.size() # batch, seq_len, embedding dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Initialize kv_cache if not provided (first step)
        if use_cache and kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=-2) # Append new keys
            v = torch.cat([kv_cache["v"], v], dim=-2) # Append new values

        # Update kv_cache if caching is enabled
        updated_kv_cache = {"k": k, "v": v} if use_cache else None
        
        # Compute attention weights
        att_weights = None
        if output_attentions:
            att_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            att_weights = F.softmax(att_weights, dim=-1)
            y = torch.matmul(att_weights, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True) ## 4 flashattention

        # Reshape back to original size
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        # Return outputs and attention weights if needed
        return y, att_weights, updated_kv_cache