import torch
import torch.nn as nn
import math

def make_window_dilation_pairs(alpha, sequence_length):
    i = 1
    pairs = []
    while i*4 <= sequence_length:
        pairs.append((i*4, i)) # window_size, dilation_rate
        i *= alpha
    return pairs 

def create_dilated_mask(row_dim, col_dim, dilation_rate, head_index=0, offset=True):
    mask = torch.zeros(row_dim, col_dim)
    start = (head_index % dilation_rate) if offset else 0
    for i in range(start, row_dim, dilation_rate):
        for j in range(start, col_dim, dilation_rate):
            # if i >= j:
            mask[i, j] = 1
    return mask

def sparseToDense(sparse_tensor, dilation_rate, head_index=0, offset=True):
    leading_dims = sparse_tensor.shape[:-2]
    s_r, s_c = sparse_tensor.shape[-2], sparse_tensor.shape[-1]
    d_r, d_c = s_r // dilation_rate, s_c // dilation_rate
    dense_tensor = torch.zeros(*leading_dims, d_r, d_c, device=sparse_tensor.device)
    
    start = (head_index % dilation_rate) if offset else 0
    for i in range(d_r):
        for j in range(d_c):
            dense_tensor[..., i, j] = sparse_tensor[..., start + i * dilation_rate, start + j * dilation_rate]
    return dense_tensor

def denseToSparse(dense_tensor, dilation_rate, head_index=0, offset=True):
    leading_dims = dense_tensor.shape[:-2]
    d_r, d_c = dense_tensor.shape[-2], dense_tensor.shape[-1]
    s_r, s_c = d_r * dilation_rate, d_c * dilation_rate
    sparse_tensor = torch.zeros(*leading_dims, s_r, s_c, device=dense_tensor.device)
    
    start = (head_index % dilation_rate) if offset else 0
    for i in range(d_r):
        for j in range(d_c):
            sparse_tensor[..., start + i * dilation_rate, start + j * dilation_rate] = dense_tensor[..., i, j]
    return sparse_tensor

class MixedDilatedAttentionDeprecated(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.alpha = config.alpha

    # attention within a window
    def dilated_attention_window(self, partial_q, partial_k, partial_v, window_size, dilation_rate, dropout_p=0.0, is_causal=False):
        device = partial_q.device
        head_index, window_size, hidden_dim = 0, partial_q.size(-2), partial_k.size(-1) # FIXME: head_index
        scale_factor = 1 / math.sqrt(hidden_dim)
        attn_bias = torch.zeros(window_size, window_size, dtype=partial_q.dtype, device=device)
    
        # generate and apply masks to q, k, and v
        mask = create_dilated_mask(window_size, hidden_dim, dilation_rate, head_index, offset=True).to(device)
        masked_q = partial_q * mask
        masked_k = partial_k * mask
        masked_v = partial_v * mask
        
        # Apply causal mask if is_causal is True
        if is_causal:
            causal_mask = torch.tril(torch.ones(window_size, window_size, dtype=torch.bool, device=device))
            attn_bias.masked_fill_(~causal_mask, float("-inf") )
        
        attn_weight = torch.matmul(masked_q, masked_k.transpose(-2, -1)) * scale_factor + attn_bias
        attn_weight = sparseToDense(attn_weight, dilation_rate, head_index).to(device)
        
        # print(attn_weight)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = denseToSparse(attn_weight, dilation_rate, head_index).to(device)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        
        output_hat = attn_weight @ masked_v
        output_hat = output_hat * mask # output masking rule
        num_row = int(attn_weight.sum(dim=-1).sum().item()) # row that has some values other than zeros
        return output_hat, attn_weight, num_row

    def forward(self, x, kv_cache=None, use_cache=False, output_attentions=False):
        B, T, C = x.size() # batch, seq_len, embedding dim (from nanogpt)
        head_dim = C // self.n_head
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hs)

        y = torch.zeros_like(x)
        denominator = []
        # wr_pairs = make_window_dilation_pairs(alpha=self.alpha, sequence_length=T)
        # wr_pairs = [(2048, 1), (4096, 2), (8192,4), (16384, 6), (32768, 12)] # paper setting
        wr_pairs = [(64,1),(128, 2), (512, 4), (1024,8), (2048, 16) ]

        # Initialize kv_cache if not provided (first step)
        if use_cache and kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=-2) # Append new keys
            v = torch.cat([kv_cache["v"], v], dim=-2) # Append new values

        # Update kv_cache if caching is enabled
        updated_kv_cache = {"k": k, "v": v} if use_cache else None
        
        for window_size, dilation_rate in wr_pairs: # multiple segment - dilation pairs
            partial_denominator = 0
            num_windows = T // window_size
            concated_output = torch.zeros_like(x)

            for i in range(num_windows): # parallel segment
                start = i * window_size
                end = start + window_size
                
                # Slice out the window for q, k, v
                partial_q = q[:, :, start:end, :]  # (B, nh, window_size, hs)
                partial_k = k[:, :, start:end, :]  # (B, nh, window_size, hs)
                partial_v = v[:, :, start:end, :]  # (B, nh, window_size, hs)
                
                window_output, attn_weight, num_row = self.dilated_attention_window(
                    partial_q, partial_k, partial_v, window_size, dilation_rate, is_causal=True
                )

                # Reshape window_output to (B, window_size, C) for placement in concated_output
                window_output = window_output.transpose(1, 2).reshape(B, window_size, C)
                concated_output[:, start:end, :] = window_output
                partial_denominator += num_row
            
            denominator.append(partial_denominator)
            y += concated_output * partial_denominator
  
        y /= sum(denominator)
        
        att_weights = None
        
        return y, att_weights, updated_kv_cache

# class Block(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.attn = MixedDilatedAttention(config)
    
#     def forward(self, x):
#         attn_output, attn_weights, updated_kv_cache = self.attn(x)
#         x = x + attn_output
#         return x, attn_weights, updated_kv_cache
    
# class Config:
#     # block_size: int = 16 # max seq_len
#     n_embd = 4
#     n_head = 1
#     alpha = 2

# config = Config()
# sequence_length = 32
# hidden_dim = config.n_embd

# x = torch.randn(1, sequence_length, hidden_dim)  # Batch size of 1
# attention_layer = MixedDilatedAttention(config)
# output = attention_layer(x)
# print(output)