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

SCALER_INIT_SA = 0.08
SCALER_INIT_MLP = 0.08

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
        
        # not kept positive, paper says perf is the same
        self.sa_scaler = Scaler(dim=config.d_model, init=SCALER_INIT_SA, scale=1/math.sqrt(config.d_model))
        self.sa = SelfAttentionMultiHead(config)

        # not kept positive, paper says perf is the same
        self.mlp_scaler = Scaler(dim=config.d_model, init=SCALER_INIT_MLP, scale=1/math.sqrt(config.d_model))
        self.mlp = MLP(config)
        
    def forward(self, X):
        # X : (B, L, D)
        # -> Y : (B, L, D)

        X = F.normalize(X + self.sa_scaler() * (F.normalize(self.sa(X), dim=-1) - X), dim=-1)
        X = F.normalize(X + self.mlp_scaler() * (F.normalize(self.mlp(X), dim=-1) - X), dim=-1)

        return X
    
    def get_empty_cache(self, batch_size):
        return (None, None)
    
class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.fc_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc_2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.fc_3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc_1.NORMALIZE = 1
        self.fc_2.NORMALIZE = 1
        self.fc_3.NORMALIZE = 1

        self.fc1_scaler = Scaler(dim=config.d_ff, init=1, scale=1) # fc1(x) is v
        self.fc3_scaler = Scaler(dim=config.d_ff, init=1, scale=1) # fc3(x) is u

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        u = self.fc3_scaler() * self.fc_3(x)
        v = self.fc1_scaler() * self.fc_1(x) * math.sqrt(self.config.d_model)

        return self.dropout(self.fc_2(F.silu(v) * u))

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.c_q = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.c_k = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.c_v = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.c_q.NORMALIZE = 1
        self.c_k.NORMALIZE = 1
        self.c_v.NORMALIZE = 1

        # todo : what about n_kv_heads != d_head?
        self.qk_scaler = Scaler(dim=self.config.d_head, init=1, scale=1/math.sqrt(self.config.d_model))

        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.c_proj.NORMALIZE = 1

        self.rotary = Rotary(config.d_head)

        #self.scale = self.config.mup_attn_mult/self.config.d_head if self.config.mup else 1/math.sqrt(self.config.d_head)
        self.scale = math.sqrt(self.config.d_head)

    def forward(self, x, cache=None):
        B, T, _ = x.size()

        # q,k,v computations
        q = self.c_q(x).view(B, T, self.config.n_heads, self.config.d_head)
        k = self.c_k(x).view(B, T, self.config.n_kv_heads, self.config.d_head)
        v = self.c_v(x).view(B, T, self.config.n_kv_heads, self.config.d_head)

        # todo : do that before rope ?
        # qk norm and rescaling
        q = self.qk_scaler() * F.normalize(q, dim=-1)
        k = self.qk_scaler() * F.normalize(k, dim=-1)

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
    
class Scaler(nn.Module):
    def __init__(self, dim, init, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * scale)
        self.forward_scale = init / scale

    def forward(self):
        return self.scale * self.forward_scale

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
