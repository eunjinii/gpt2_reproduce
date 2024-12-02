import torch
import torch.nn as nn
import math
import time
from torch.nn import functional as F

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

class DifferentialFlashAttention(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head // 2 # head_dim splitted by two for dual-query mechanism
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        
        # self.rms_norm = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, kv_cache=None, use_cache=False, output_attentions=False):
        B, N, D = x.size()  # batch size, sequence length, embedding dimension
        qkv = self.c_attn(x)  # combined query, key, value projection
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, N, 2 * self.n_head, self.head_dim) # torch.Size([B, N, 2H, d])
        k = k.view(B, N, 2 * self.n_head, self.head_dim) # torch.Size([B, N, 2H, d])
        v = v.view(B, N, self.n_head, 2 * self.head_dim) # torch.Size([B, N, H, 2d])
        
        # Split the last dimension into two components for dual-query mechanism
        q = q.reshape(B, N, self.n_head, 2, self.head_dim)
        k = k.reshape(B, N, self.n_head, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1] # torch.Size([B, N, H, 2, d])
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1] # torch.Size([B, N, H, 2, d])
        
        att_weights1 = F.scaled_dot_product_attention(q1, k1, v) # torch.Size([B, N, H, 2d])
        att_weights2 = F.scaled_dot_product_attention(q2, k2, v) # torch.Size([B, N, H, 2d])

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        y = att_weights1 - lambda_full * att_weights2
        
        # y = self.rms_norm(y)
        y = y * (1 - self.lambda_init)
        y = y.reshape(B, N, self.n_head * 2 * self.head_dim)
        y = self.out_proj(y) # torch.Size([B, N, D])
        
        att_weights, updated_kv_cache = None, None

        return y, att_weights, updated_kv_cache