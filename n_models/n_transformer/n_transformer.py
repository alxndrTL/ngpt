import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.rope import Rotary, apply_rotary_emb

"""
caching is WIP
"""

#todo : mettre des optional la ou on peut
@dataclass
class TransformerConfig:
    d_model: int # D or d_model in comments
    n_layers: int
    n_heads: int
    max_len: int # maximum sequence length (for positional embedding, super attn and mask if no FA)
    dropout: float = 0.
    norm_eps: float = 1e-5
    base_std: float = 0.02
    
    d_ff: int = None
    n_kv_heads: Optional[int] = None # None=n_heads is MHA, 1 is MQA (multi query), in between is GQA (grouped)

    diff_transformer: bool = False

    pos_emb: str = "absolute" # absolute, rope
    rope_theta: float = 10000

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    flash: bool = True

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be a multiple of n_heads"
        self.d_head = self.d_model // self.n_heads

        self.n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0, "number of kv heads must divide the number of heads"
        self.kv_rep = self.n_heads // self.n_kv_heads

        if self.d_ff is None:
            self.d_ff = 4*self.d_model

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width
            self.mup_attn_mult = math.sqrt(self.d_head) # base_d_head=d_head (kept constant)

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        if self.config.pos_emb == "absolute":
            self.PE = nn.Embedding(config.max_len, config.d_model)

        self.layers = nn.ModuleList([DecoderLayer(config, i+1) for i in range(config.n_layers)])
        
        self.in_dropout = nn.Dropout(config.dropout)

    def forward(self, X, seq_pos=0):
        # X : (B, L, D)

        # Y : (B, L, D)

        _, T, _ = X.size()

        if self.config.pos_emb == "absolute":
            pos_emb = self.PE(torch.arange(seq_pos, seq_pos+T, dtype=torch.long, device=X.device))
            X = self.in_dropout(X + pos_emb)
        else:
            X = self.in_dropout(X)

        for layer in self.layers:
            X = layer(X) # (B, L, d_model)
        
        return X

class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig, depth: int):
        super().__init__()

        self.sa_scale = (1 / math.sqrt(2 * config.n_layers))

        self.attention_norm = Norm(config.d_model, config.norm_eps, config.mup) # to define
        if config.diff_transformer:
            self.sa = SelfDifferientialAttentionMultiHead(config, depth)
        else:
            self.sa = SelfAttentionMultiHead(config)
        self.mlp_norm = Norm(config.d_model, config.norm_eps, config.mup) # to define
        self.mlp = MLP(config)
        
    def forward(self, X):
        # X : (B, L, D)
        # -> Y : (B, L, D)

        X = X + self.sa_scale * self.sa(self.attention_norm(X))
        X = X + self.mlp(self.mlp_norm(X))

        return X
    
    def get_empty_cache(self, batch_size):
        return (None, None)
    
class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.fc_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc_2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.fc_3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc_2(F.silu(self.fc_1(x)) * self.fc_3(x)))

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.c_attn = nn.Linear(config.d_model, (config.n_heads + 2 * config.n_kv_heads) * self.config.d_head, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = Rotary(config.d_head)

        #self.scale = self.config.mup_attn_mult/self.config.d_head if self.config.mup else 1/math.sqrt(self.config.d_head)
        self.scale = 1/math.sqrt(self.config.d_head)

    def forward(self, x, cache=None):
        B, T, _ = x.size()

        # q,k,v computations
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.config.n_heads * self.config.d_head, self.config.n_kv_heads * self.config.d_head, self.config.n_kv_heads * self.config.d_head], dim=-1)
        q, k, v = map(lambda t: t.view(B, T, -1, self.config.d_head), (q, k, v)) # (B, T, n_heads, d_head)

        # RoPE
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # (B, n_heads, T, d_head)

        # GQA : expand K and V to compute standard attention
        k = repeat_kv(k, self.config.kv_rep)
        v = repeat_kv(v, self.config.kv_rep)

        # attention computation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None), scale=self.scale)
        y = y.transpose(1, 2).contiguous().view(B, T, self.config.d_model)
        
        # output projection
        y = self.c_proj(y)
        return y
    
# todo : handle n_kv_heads!=n_heads
def lambda_init_fn(depth): # depth in [|1, L|]
    return 0.8 - 0.6 * math.exp(-0.3 * (depth-1))

class SelfDifferientialAttentionMultiHead(nn.Module):
    def __init__(self, config: TransformerConfig, depth: int):
        super().__init__()
        self.config = config

        self.n_heads = config.n_heads//2
        self.d_head = config.d_head
        self.kv_rep = 1
        
        self.c_attn = nn.Linear(config.d_model, (2 * self.n_heads + 2 * 2 * self.n_heads) * self.d_head, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = Rotary(self.d_head)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.d_head, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.d_head, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.d_head, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.d_head, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.subln = RMSNorm(dim=2*self.d_head, eps=1e-5, use_mup=False)

        self.scale = 1/math.sqrt(self.config.d_head)

    def forward(self, x, cache=None):
        B, T, _ = x.size()

        # q,k,v computations
        qkv = self.c_attn(x) # (B, L, n_heads*d_head+2*n_kv_heads*d_head)
        q, k, v = qkv.split([2 * self.n_heads * self.d_head, 2 * self.n_heads * self.d_head, 2 * self.n_heads * self.d_head], dim=-1)
        q, k = map(lambda t: t.view(B, T, -1, self.d_head), (q, k)) # (B, T, 2*n_heads, d_head)
        v = v.view(B, T, self.n_heads, 2 * self.d_head)

        # RoPE
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # q,k : (B, 2*n_heads, T, d_head), v : (B, n_heads, T, 2*d_head)
        q = q.reshape(B, 2, self.n_heads, T, self.d_head)
        k = k.reshape(B, 2, self.n_heads, T, self.d_head)

        q1, q2 = q[:, 0], q[:, 1] # (B, n_heads, T, d_head)
        k1, k2 = k[:, 0], k[:, 1] # (B, n_heads, T, d_head)

        attn1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=(cache is None), scale=self.scale)
        attn2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=(cache is None), scale=self.scale)
        # attn12 : (B, n_heads, T, 2*d_head)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn = attn1 - lambda_full * attn2 # (B, n_heads, T, 2*d_head)

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)

        attn = attn.transpose(1, 2).contiguous().view(B, T, self.config.d_model)
        y = self.c_proj(attn)
        return y

# taken from modeling_jamba.py (jamba official implementation)
# the same as the one in llama2.c model.py, but dim of repeat is 1 instead of 2
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim).
    """

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
