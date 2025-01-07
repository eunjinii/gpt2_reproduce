import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()  # Clear the cache

class DilatedAttention2(nn.Module):
    def __init__(self, config, segment_size, dilation_rate):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        
        # Linear Projections
        self.proj_q = nn.Linear(config.n_embd, config.n_embd, bias=False).to(device)
        self.proj_k = nn.Linear(config.n_embd, config.n_embd, bias=False).to(device)
        self.proj_v = nn.Linear(config.n_embd, config.n_embd, bias=False).to(device)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False).to(device)
        
        self.norm = nn.LayerNorm(config.n_embd).to(device)
        
    def forward(self, x, kv_cache=None, use_cache=False, output_attentions=False):
        B, N, D = x.size()
        device = x.device
        
        assert N % self.segment_size == 0, f"N: {N}, segment_size: {self.segment_size}"
        assert self.segment_size % self.dilation_rate == 0
        
        # Sparsify
        x = x.view(B, N // self.segment_size, self.segment_size, D)
        x = x[:, :, :: self.dilation_rate, :]
        q, k, v = map(self.norm, (self.proj_q(x), self.proj_k(x), self.proj_v(x))) # q,k,v: torch.Size([B, num_segments, segment_size // dilation_rate, D])
        
        # TODO: Implement cache
        # TODO: Implement shifting positions
        
        # All gather
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # torch.Size([B, num_segments, segment_size // dilation_rate, D])
        y = y.reshape(B, -1, D) # torch.Size([B, N // dilation_rate, D])
        y_full = torch.zeros(B, N, D, device=y.device, dtype=y.dtype)
        y_full[:, ::self.dilation_rate, :] = y
        y_full = self.out_proj(y_full)
        
        att_weights, updated_kv_cache = None, None
        
        return y_full, att_weights, updated_kv_cache

class MixedDilatedAttentionDeprecated(nn.Module):
    def __init__(self, config, wr_pairs):
        super().__init__()
        self.config = config
        self.wr_pairs = wr_pairs
        self.dilated_attn = nn.ModuleList()
        for segment_size, dilation_rate in self.wr_pairs:
            self.dilated_attn = nn.ModuleList(
                [DilatedAttention2(self.config, segment_size, dilation_rate) for segment_size, dilation_rate in self.wr_pairs]
            )

        from causal_self_attention import CausalSelfAttention
        self.self_attn = CausalSelfAttention(config)

    def forward(self, x, kv_cache=None, use_cache=False, output_attentions=False):
        N = x.size(1)
        y = None
        
        is_dilated = False
        for segment_size, _ in self.wr_pairs:
            if N % segment_size == 0:
                is_dilated = True
            else:
                is_dilated = False
                break
        
        if is_dilated:
            for block in self.dilated_attn:
                output, _, _ = block(x)
                y = output if y is None else y + output
        else:
            output, _, _ = self.self_attn(x)
            y = output
        
        att_weights, updated_kv_cache = None, None
        return y, att_weights, updated_kv_cache
