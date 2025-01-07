import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchscale.architecture.config import DecoderConfig
from torchscale.component.dilated_attention import DilatedAttention as di_attn

torch.cuda.empty_cache()  # Clear the cache
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DilatedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd).to(device)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        decoder_config = DecoderConfig(
            vocab_size=50304,
            segment_length='[512,1024,2048]',
            dilated_ratio='[1,2,4]',
            flash_attention=True,
        )
        self.attn = di_attn(
            args = decoder_config,
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=0.1,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=False,
        )

    def forward(self, x, kv_cache=None, use_cache=False, output_attentions=False):
        B, T, C = x.size() # batch, seq_len, embedding dim
        qkv = self.c_attn(x)
        query, key, value = qkv.split(self.n_embd, dim=2)

        y, _ = self.attn(
            query,
            key,
            value,
            incremental_state=None,
            key_padding_mask=None,
            attn_mask=None,
            rel_pos=None, # ?
            is_first_step=False, # ?
            is_causal=True,
        )
        
        return y 