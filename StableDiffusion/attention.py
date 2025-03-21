import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
  def __init__(self,
               n_heads,
               d_embed,
               in_proj_bias=True,
               out_proj_bias=True):
    super().__init__()
    self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
    self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
    self.n_heads = n_heads
    self.d_head = d_embed // n_heads

  def forward(self, x, causal_mask=False):
    #x : [B, S, D]
    input_shape = x.shape
    batch_size, seq_len, d_embed = input_shape
    interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
    qkv = self.in_proj(x)
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    q = q.view(interim_shape).transpose(1, 2)   #[B, n_heads, S, d_head]
    k = k.view(interim_shape).permute(2, 3, 1)  #[B, n_heads, d_head, S]
    v = v.view(interim_shape).transpose(1, 2)  #[B, n_heads, S, d_head]

    attention_weights = q @ k / math.sqrt(self.d_head)
    if causal_mask:
      mask = torch.ones_like(attention_weights, dtype=torch.bool).triu(1)
      attention_weights.masked_fill_(mask, -torch.inf)
    attention_scores = F.softmax(attention_weights, dim=-1)
    output = attention_scores @ v   #[B, n_heads, S, d_head]
    output = output.transpose(2, 1) #[B, S, n_heads, d_head]
    output = output.reshape(input_shape)
    return self.out_proj(output)


class CrossAttention(nn.Module):
  def __init__(self, 
               n_heads, 
               d_embed, #query dim
               d_cross, #key, value dim
               in_proj_bias=True,
               out_proj_bias=True):
    super().__init__()
    self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
    self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
    self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
    self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
    self.n_heads = n_heads
    self.d_head = d_embed // n_heads

  def forward(self, x, y): #x=query=latent=[B, Seq_q, q_dim], y=key, values, context/prompt = [B, Seq_kv, kv_dim]
    input_shape = x.shape 
    batch_size, sequence_length, d_embed = input_shape
    interim_shape = (batch_size, -1, self.n_heads, self.d_head)

    q = self.q_proj(x)
    k = self.k_proj(y)
    v = self.v_proj(y)

    q = q.view(interim_shape).transpose(1, 2)   #[B, S, n_heads, d_head]-[B, n_heads, S, d_head]
    k = k.view(interim_shape).permute(2, 3, 1)  #[B, n_heads, d_head, S]
    v = v.view(interim_shape).transpose(1, 2)   #[B, n_heads, S, d_head]

    attention_weights = q @ k / math.sqrt(self.d_head)
    attention_scores = F.softmax(attention_weights, dim=-1)
    output = attention_scores @ v               #[B, n_heads, S, d_head]
    output = output.transpose(2, 1)             #[B, S, n_heads, d_head]
    output = output.reshape(input_shape)
    return self.out_proj(output)

