import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from attention.norms import RMSNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()  # Clear the cache

class LinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        
        self.relu = nn.ReLU()
    
    def forward(self, x, kv_cache=None, use_cache=False, output_attentions=False):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).permute(0, 2, 1, 3)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).permute(0, 2, 3, 1)  # (B, n_head, head_dim, T)
        v = v.view(B, T, self.n_head, self.head_dim).permute(0, 2, 1, 3)  # (B, n_head, T, head_dim)
        
        q = self.relu(q)
        k = self.relu(k)
        
        # numerator
        kv = torch.matmul(k, v)  # (B, n_head, head_dim, head_dim)
        qkv_weighted_sum = torch.matmul(q, kv)  # (B, n_head, T, head_dim)
        
        # denominator
        k_sum = torch.sum(k, dim=-1, keepdim=True)  # (B, n_head, head_dim, 1)
        qk_sum = torch.matmul(q, k_sum)  # (B, n_head, T, 1)

        y = qkv_weighted_sum / (qk_sum + 1e-6)  # (B, n_head, T, head_dim)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)  # (B, T, C)

        att_weights, updated_kv_cache = None, None
        return y, att_weights, updated_kv_cache
