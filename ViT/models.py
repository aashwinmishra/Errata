import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
  """
  Splits given images into patches, 
  Embedds each patch to a vector.
  Parameters:
    img_size: Size of the (square) input image
    patch_size: Size of the (square) patches
    in_channels: Number of image channels
    embed_dim: Dimensionality of embedded vectors
  Attributes:
    n_patches: Number of patches
    proj: Convolutional Layer caarrying out splitting and embedding.

  """
  def __init__(self, 
               img_size: int, 
               patch_size: int, 
               in_channels: int=3, 
               embed_dim: int=768):
    super().__init__()
    self.img_size = img_size 
    self.patch_size = patch_size 
    self.n_patches = (img_size // patch_size)**2

    self.proj = nn.Conv2d(in_channels=in_channels, 
                          out_channels=embed_dim, 
                          kernel_size=patch_size, 
                          stride=patch_size)
    
  def forward(self, x):
    return self.proj(x).flatten(2).transpose(1,2)


class Attention(nn.Module):
  """
  Basic MHA module for the Transformer block.
  Parameters:
    dim: Embedding dimension.
    n_heads: Number of attention heads for MHA.
    qkv_bias: If there is a bias in the qkv projections.
    attn_p: Dropout rate applied to qkv tensors
    proj_p: Dropout rate applied to the output
  Attributes:
    scale: the normalizing factor for the <Q, K>
    qkv: Linear projection for the qkv tensors
    proj: Linear projection for the output
    attn_drop: Dropout layer
    proj_drop: Dropout layer
  """
  def __init__(self, 
               dim: int, 
               n_heads: int=12, 
               qkv_bias: bool=True, 
               attn_p: float=0.0, 
               proj_p: float=0.0):
    super().__init__()
    self.dim = dim
    self.n_heads = n_heads
    self.head_dim = dim // n_heads
    self.scale = (self.head_dim)**0.5
    self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
    self.attn_drop = nn.Dropout(p=attn_p)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(p=proj_p)

  def forward(self, x):
    batch_size, sequence_length, dim = x.shape
    qkv = self.qkv(x)
    Q, K, V = torch.chunk(qkv, chunks=3, dim=-1)

    Q = Q.reshape((batch_size, sequence_length, self.n_heads, self.head_dim))
    K = K.reshape((batch_size, sequence_length, self.n_heads, self.head_dim))
    V = V.reshape((batch_size, sequence_length, self.n_heads, self.head_dim))

    Q = Q.transpose(1,2)
    K = K.permute(0, 2, 3, 1)
    V = V.transpose(1,2)

    attn_weights = nn.functional.softmax(Q @ K/self.scale, dim=-1)
    attn_weights = self.attn_drop(attn_weights)
    attention = attn_weights @ V 
    attention = attention.transpose(1,2)
    attention = attention.reshape((batch_size, sequence_length, dim))
    output = self.proj(attention)
    return self.proj_drop(output)


class MLP(nn.Module):
  """
  MLP module for the transformer block.
  Parameters:
    in_features: dim of input
    hidden_features: dim of hidden layer
    out_features: dim of output
    p: dropout rate
  Attributes:
    fc: first linear layer
    act: GeLU activation
    fc2: second linear layer
    drop: Dropout layer
  """
  def __init__(self, 
               in_features: int=768, 
               hidden_features: int=3072, 
               out_features: int=768, 
               p:float=0.1):
    super().__init__()
    self.fc = nn.Linear(in_features, hidden_features)
    self.drop = nn.Dropout(p)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(hidden_features, out_features)

  def forward(self, x):
    x = self.fc(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    return self.drop(x)


class Block(nn.Module):
  """
  Transformer block for the ViT.
  Parameters:
    dim: Embedding dim
    n_heads: Number of heads in MHA
    mlp_ratio: Dimensionality of hidden layer in MLP, mlp_ratio*input_dim
    qkv_bias: Whether to include bias in Q, K, V projections
    p: Dropout Prob
    attn_p: Dropout prob for attention 
  Attribute:
    norm1: Layer norm
    norm2: Layer norm
    attn: attention module
    mlp: MLP block
  """
  def __init__(self, 
               dim: int=768, 
               n_heads: int=12, 
               mlp_ratio: int=4, 
               qkv_bias: bool=True, 
               p: float=0.0, 
               attn_p: float=0.0):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim, eps=1e-6)
    self.attn = Attention(dim, n_heads, qkv_bias, attn_p, p)
    self.norm2 = nn.LayerNorm(dim, eps=1e-6)
    self.mlp = MLP(dim, mlp_ratio*dim, dim, p)

  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
  """
  Basic Implimentation of the ViT.
  Parameters:
    img_size: Dimension of the Square Image
    patch_size: Dimension of the Square Patch
    in_channels: Number of channels
    n_classes: Number of classes
    embed_dim: Dimensionality of token embeddings
    depth: Number of Transformer blocks
    n_heads: Number of attention heads in MHA
    mlp_ratio: Dimensionality of hidden layer in MLP, mlp_ratio*input_dim
    qkv_bias: Whether to include bias in Q, K, V projections
    p: Dropout Prob
    attn_p: Dropout prob for attention 

  Attributes:
    patch_embed: Instance of PatchEmbedding module
    cls_token: learnable parameter representing the first token in the sequence, [embed_dim,]
    pos_emb: Positional Embeddings for the inputs to the transformer, [n_patches+1, embed_dim]
    pos_drop: Dropout Layer
    blocks: ModuleList of Transformer blocks
    norm: Layer normalization
    head: final output layer
  """
  def __init__(self, 
               img_size: int, 
               patch_size: int, 
               in_channels: int, 
               n_classes: int, 
               embed_dim: int, 
               depth: int, 
               n_heads: int, 
               mlp_ratio: int, 
               qkv_bias: bool, 
               p: float, 
               attn_p: float):
    super().__init__()
    self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
    self.cls_token = nn.Parameter(torch.ones(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.ones(1, self.patch_embed.n_patches + 1, embed_dim))
    self.pos_drop = nn.Dropout(p)
    self.blocks = nn.ModuleList([Block(embed_dim, n_heads, mlp_ratio, qkv_bias, p, attn_p) for _ in range(depth)])
    self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
    self.head = nn.Linear(embed_dim, n_classes)

  def forward(self, x):
    batch_size = x.shape[0]
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = x + self.pos_embed
    x = self.pos_drop(x)
    for blk in self.blocks:
      x = blk(x)
    x = self.norm(x)
    cls_token_final = x[:, 0]
    return self.head(cls_token_final)

