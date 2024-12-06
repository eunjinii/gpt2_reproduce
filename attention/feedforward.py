import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from attention.norms import RMSNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()  # Clear the cache

class MLPFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class MixFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        # self.inverted_conv = nn.Conv1d(config.n_embd, self.hidden_dim * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            config.n_embd,
            config.n_embd,
            kernel_size=3,
            padding=1,
            groups=config.n_embd,
        )
        self.act = nn.SiLU()
        # self.act = nn.ReLU()
        self.pointwise_conv = nn.Conv1d(
            config.n_embd, 
            config.n_embd, 
            kernel_size=1
        )
        self.norm = RMSNorm(config.n_embd)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)

        # Gating 
        # x, gate = torch.chunk(x, 2, dim=1)  # Split into (x, gate)
        gate = self.act(x)
        x = x * gate
        
        x = self.pointwise_conv(x) 
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.dropout(x)
        return x