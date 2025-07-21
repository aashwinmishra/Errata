"""
Model definitions for vision classification tasks.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
  """
  Creates patch embeddings to enable images to work with transformer blocks.

  Attributes:
    patcher: Breaks down batch of images into flattened embedded patches.
    class_token: Prepends learnable class token.
    positional_encoding: Learnable positional encodings for order.
  """
  def __init__(self, 
               in_channels: int, 
               img_size: int, 
               patch_size: int, 
               embed_dim: int):
    """
    Initializes an instance of the patch embedding layer.

    Args:
      inn_channels: Number of channels in the input images.
      img_size: Size of square input images.
      patch_size: Size of square patches.
      embed_dim: Dimension of embedding space.
    """
    assert img_size % patch_size == 0, "Image Size must be exactly divisible by Patch Size"
    super().__init__()
    self.patcher = nn.Conv2d(in_channels=in_channels,
                             out_channels=embed_dim,
                             kernel_size=patch_size,
                             stride=patch_size,
                             padding=0)
    self.class_token = nn.Parameter(torch.rand(1, 1, embed_dim))
    self.positional_encoding = nn.Parameter(torch.rand(1, 1 + img_size**2 // patch_size**2, embed_dim))

  def forward(self, x: torch.tensor) -> torch.tensor:
    patches = self.patcher(x).flatten(-2, -1).permute(0, 2, 1)
    return torch.cat((self.class_token.repeat(patches.shape[0], 1, 1), patches), dim=1) + self.positional_encoding


class LayerNorm(nn.Module):
  """
  Normalizes along the feature/channel dimension, ie, Layer Norm.

  Attributes:
    eps: Factor added to denominator for circumventing the divide by zero error.
    scale: Learnable parameter to scale along each feature.
    shift: Learnable parameter to shift along each feature.
  """
  def __init__(self, 
               normalized_shape: int, 
               eps:float=0.00001):
    super().__init__()
    self.eps = eps 
    self.scale = nn.Parameter(torch.ones(normalized_shape))
    self.shift = nn.Parameter(torch.zeros(normalized_shape))

  def forward(self, x: torch.tensor) -> torch.tensor:
    return (x - torch.mean(x, dim=-1, keepdim=True)) / torch.sqrt(torch.var(x, dim=-1, keepdim=True) + self.eps) * self.scale + self.shift


class MultiHeadAttention(nn.Module):
  """
  Multi-Head Self-Attention Layer, without masking for ViT.

  Attributes:
    Wq: Linear Projection for the query.
    Wk: Linear Projection for the key.
    Wv: Linear Projection for the value.
    Wo: Linear Projection for the output.
    num_heads: Number of attention heads.
    head_dim: Dimensionality of each head.
  """
  def __init__(self, 
               in_dim: int, 
               out_dim: int, 
               num_heads: int, 
               qkv_bias:bool=False):
    """
    Initializes an instance of the MHSA layer object.

    Args:
      in_dim: Dimensionality of the input embedding.
      out_dim: Dimensionality of the output.
      num_heads: Nnumber of attention heads.
      dropout: Dropout rate.
      qkv_bias: If to apply bias on the projections.
    """
    assert out_dim % num_heads == 0, "Embedding dim should be a multiple of head dim"
    super().__init__()
    self.Wq = nn.Linear(in_dim, out_dim, bias=qkv_bias)
    self.Wk = nn.Linear(in_dim, out_dim, bias=qkv_bias)
    self.Wv = nn.Linear(in_dim, out_dim, bias=qkv_bias)
    self.Wo = nn.Linear(out_dim, out_dim, bias=qkv_bias)
    self.num_heads = num_heads
    self.head_dim = out_dim // num_heads

  def forward(self, x: torch.tensor) -> torch.tensor:
    Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
    b, n, dm = Q.shape
    Q = torch.reshape(Q, (b, n, self.num_heads, self.head_dim)).permute(0, 2, 1, 3) #[B, h, N, dh]
    K = torch.reshape(K, (b, n, self.num_heads, self.head_dim)).permute(0, 2, 3, 1) #[B, h, dh, N]
    V = torch.reshape(V, (b, n, self.num_heads, self.head_dim)).permute(0, 2, 1, 3) #[B, h, N, dh]
    attention_scores = Q @ K / self.head_dim**0.5
    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention = attention_weights @ V 
    attention = attention.permute(0, 2, 1, 3).reshape(b, n, -1)
    return self.Wo(attention)


class GeLU(nn.Module):
  """
  Class defining the Gaussian Error Linear Unit activation.
  """
  def __init__(self):
    super().__init__()

  def forward(self, x: torch.tensor) -> torch.tensor:
    return 0.5 * x * (1.0 + torch.tanh((2/torch.pi)**0.5 * \
     (x + 0.044715*torch.pow(x, 3))))
    

class FeedForward(nn.Module):
  """
  Class defining the MLP sub-section of the Transformer Block for the ViT.

  Attributes:
    layers: Sequential modules defining the layers in the MLP.
  """
  def __init__(self, 
               embed_dim: int, 
               mlp_size: int, 
               dropout: float):
    """
    Initializes an instance of the MLP sub-section object. 

    Args:
      embed_dim: Embedding dimension, input & output dimension for the MLP.
      mlp_size: Intermediate/hidden dimension for the MLP.
      dropout: dropout parameter for the MLP block.
    """
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(embed_dim, mlp_size),
        GeLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_size, embed_dim),
        nn.Dropout(dropout)
    )

  def forward(self, x: torch.tensor) -> torch.tensor:
    return self.layers(x)

  
class TransformerBlock(nn.Module):
  def __init__(self, 
               in_dim: int, 
               out_dim: int, 
               embed_dim: int, 
               num_heads: int, 
               mlp_size: int,
               dropout: float):
    super().__init__()
    self.attention = MultiHeadAttention(in_dim, out_dim, num_heads)
    self.MLP = FeedForward(embed_dim, mlp_size, dropout)
    self.norm1 = LayerNorm(embed_dim)
    self.norm2 = LayerNorm(embed_dim)

  def forward(self, x: torch.tensor) -> torch.tensor:
    skip = x 
    x = self.norm1(x)
    x = self.attention(x)
    x = x + skip 
    skip = x 
    x = self.norm2(x)
    x = self.MLP(x)
    return x + skip 


class ViT_Base(nn.Module):
  def __init__(self, 
               in_channels: int, 
               img_size: int, 
               patch_size: int, 
               num_layers: int, 
               num_heads: int,
               embed_dim: int, 
               mlp_size: int, 
               mlp_dropout: float, 
               num_classes: int):
    super().__init__()
    self.PatchEmbedding = PatchEmbedding(in_channels, img_size, patch_size, embed_dim)
    self.trf_blocks = nn.Sequential(
        *[TransformerBlock(in_dim=embed_dim, 
                           out_dim=embed_dim, 
                           embed_dim=embed_dim, 
                           num_heads=num_heads, 
                           mlp_size=mlp_size,
                           dropout=mlp_dropout) for _ in range(num_layers)]
    )
    self.head = nn.Sequential(
        LayerNorm(embed_dim),
        nn.Linear(embed_dim, num_classes)
    )

  def forward(self, x: torch.tensor) -> torch.tensor:
    x = self.PatchEmbedding(x)
    x = self.trf_blocks(x)
    return self.head(x[:, 0, :])




class TinyVGG(nn.Module):
  """
  Reduced scale version of VGG model to classifiy images.

  Attributes:
    representation: Convolutional section carrying out representation learning.
    classifier: Densely connected section to translate representations into classes.
  """

  def __init__(self,
               in_channels: int,
               num_classes: int,
               input_shape: int):
    """Initializes a new TinyVGG model object.

    Args:
      in_channels: Number of channels in the input images.
      num_classes: Number of classes in the output.
      input_shape: Dimension shape of the square input images
    """
    super().__init__()
    self.representation = nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1), #[n,n,16]
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), #(n/2, n/2,16)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(), #[n/2 * n/2 * 16]
        nn.Linear(int(input_shape * input_shape * 16 / 4), num_classes)
    )

  def forward(self, x: torch.tensor) -> torch.tensor:
    return self.classifier(self.representation(x))

