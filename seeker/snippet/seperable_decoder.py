#date: 2022-02-08T17:05:39Z
#url: https://api.github.com/gists/da8c21a3931997902173767b3701d31b
#owner: https://api.github.com/users/darbyhaller

# %%
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from copy import deepcopy

def mask(sq, sk):
    assert sk >= sq
    m = torch.triu(torch.full((sk, sk), float('-inf')), diagonal=1)
    return m[-sq:]

def get_kv(self, cfg, x):
    _, w_k, w_v = self.in_proj_weight.chunk(3)
    _, b_k, b_v = self.in_proj_bias.chunk(3)

    k = F.linear(x, w_k, b_k)
    k = k.contiguous().view(k.shape[0], cfg.n_heads, int(cfg.d_model / cfg.n_heads)).transpose(0, 1)

    v = F.linear(x, w_v, b_v)
    v = v.contiguous().view(v.shape[0], cfg.n_heads, int(cfg.d_model / cfg.n_heads)).transpose(0, 1)

    return k, v

def efficient_attention(self, cfg, query, static_k, static_v):
    dummy_kv = torch.zeros(0, 1, cfg.d_model, device=query.device)

    attn_output, _ = F.multi_head_attention_forward(
        query, dummy_kv, dummy_kv, cfg.d_model, cfg.n_heads, self.in_proj_weight, self.in_proj_bias,
        self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias,
        attn_mask=mask(query.shape[0], static_k.shape[1]),
        use_separate_proj_weight=True, static_k=static_k, static_v=static_v,
        q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight)

    return attn_output

def efficient_layer(self, cfg, x, static_k, static_v):
    attn_output = efficient_attention(self.mha2, cfg, x, static_k, static_v)
    attn_output = self.dropout1(attn_output)
    x = self.norm1(x + attn_output)
    x = self.norm2(x + self._ff_block(x))
    return x

class SeparableDecoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.hparams = cfg
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, 4 * cfg.d_model, dropout=cfg.dropout),
            num_layers=cfg.n_layers
        )
        for mod in self.decoder.layers:
            mha2 = deepcopy(mod.self_attn)
            mha2._qkv_same_embed_dim = False
            mha2.q_proj_weight = Parameter(mod.self_attn.in_proj_weight[:cfg.d_model])
            # neither of these should be used ever
            mha2.k_proj_weight = Parameter(mod.self_attn.in_proj_weight[cfg.d_model:2 * cfg.d_model])
            mha2.v_proj_weight = Parameter(mod.self_attn.in_proj_weight[2 * cfg.d_model:])
            mod.mha2 = mha2
    
    def forward(self, x, k1s=None, v1s=None):
        cfg = self.hparams

        if k1s is None:
            dummy_kv = torch.zeros(cfg.n_heads, 0, int(cfg.d_model / cfg.n_heads))
            k1s = [dummy_kv for _ in range(cfg.n_heads)]
            v1s = [dummy_kv for _ in range(cfg.n_heads)]

        k2s, v2s = [], []
        for i, layer in enumerate(self.decoder.layers):
            k2, v2 = get_kv(layer.self_attn, cfg, x)
            k2s.append(k2); v2s.append(v2)
            k = torch.cat([k1s[i], k2s[i]], dim=1)
            v = torch.cat([v1s[i], v2s[i]], dim=1)
            x = efficient_layer(layer, cfg, x, k, v)
        return x, k2s, v2s

if __name__ == "__main__":
    cfg = Namespace(d_model=768, n_heads=4, dropout=0, n_layers=12, seq_len=1000)
    decoder = SeparableDecoder(cfg)
    embedding = nn.Embedding(cfg.seq_len, cfg.d_model)
    tokens12 = torch.arange(0, cfg.seq_len, dtype=torch.long)[:, None]
    in12 = embedding(tokens12)

    # %%
    for i in range(10):
        out12 = decoder.decoder(in12, mask=mask(cfg.seq_len, cfg.seq_len))
    # %%
    split = int(.9 * cfg.seq_len)
    out1b, k1s, v1s = decoder(in12[:split])
    for i in range(10):
        out2b, k2s, v2s = decoder(in12[split:], k1s, v1s)
    # %%
    print((out12 - torch.cat([out1b, out2b])).norm())
