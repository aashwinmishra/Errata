import torch
import torch.nn as nn


class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])
    self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )
    self.final_norm = LayerNorm(cfg["embed_dim"])
    self.out_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape 
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
    x = tok_embeds + pos_embeds
    x = self.drop_emb(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits


class LayerNorm(nn.Module):
  def __init__(self, normalized_shape: int=768, eps=1e-5):
    super().__init__()
    self.eps = eps
    self.scale = nn.Parameter(torch.ones(normalized_shape))
    self.shift = nn.Parameter(torch.zeros(normalized_shape))

  def forward(self, x):
    return self.scale * (x - torch.mean(x, dim=-1, keepdim=True)) / \
    torch.sqrt(torch.var(x, dim=-1, keepdim=True) + self.eps) + self.shift


class GeLU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (1.0 + torch.tanh((2/torch.pi)**0.5 * \
     (x + 0.044715*torch.pow(x, 3))))
    

class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(cfg["embed_dim"], 4*cfg["embed_dim"]),
        GeLU(),
        nn.Linear(4*cfg["embed_dim"], cfg["embed_dim"])
    )

  def forward(self, x):
    return self.layers(x)


class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out,
               context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert (d_out % num_heads == 0), \
    "d_out must be divisible by num_heads"
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.mask = torch.triu(torch.ones(context_length, context_length),diagonal=1)

  def forward(self, x):
    b, num_tokens, d_in = x.shape
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)
    attn_scores = queries @ keys.transpose(2, 3)
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    attn_scores.masked_fill_(mask_bool, -torch.inf)
    attn_weights = torch.softmax(
    attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)
    context_vec = (attn_weights @ values).transpose(1, 2)
    context_vec = context_vec.reshape(b, num_tokens, self.d_out)
    context_vec = self.out_proj(context_vec)
    return context_vec


class TransformerBlock(nn.Module):
  def __init__(self, cfg) -> None:
    super().__init__()
    self.att = MultiHeadAttention(d_in=cfg["embed_dim"],
                                  d_out=cfg["embed_dim"],
                                  context_length=cfg["context_length"],
                                  dropout=cfg["drop_rate"],
                                  num_heads=cfg["n_heads"],
                                  qkv_bias=cfg["qkv_bias"])
    self.ff = FeedForward(cfg)
    self.norm1 = LayerNorm(cfg["embed_dim"])
    self.norm2 = LayerNorm(cfg["embed_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

  def forward(self, x):
    skip = x 
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + skip 
    skip = x 
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + skip 
    return x

    
