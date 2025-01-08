import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchscale.architecture.config import DecoderConfig
from torchscale.component.dilated_attention import DilatedAttention as di_attn
from einops import rearrange

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
            segment_length='[256,512,1024,2048,4096]',
            dilated_ratio='[1,2,4,8,16]',
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

    def forward(
        self,
        x,
        incremental_state = None, # Originally kv_cache, renamed for compatibility with torchscale
        use_cache: bool = False, # Not used in Dilated Attention
        output_attentions: bool = False, 
        is_first_step: bool = False, # Added for compatibility with torchscale
    ):
        """
           Wrapper for the Dilated Attention module 
        """
        y, _ = self.attn(
            query=x,
            key=x,
            value=x,
            incremental_state=None, # Fixme
            is_first_step=is_first_step,
            is_causal=True,
        )
        
        return y, None