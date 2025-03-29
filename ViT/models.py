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



