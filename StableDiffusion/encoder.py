import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
  def __init__(self):
    super().__init__(
        nn.Conv2d(3, 128, kernel_size=3, padding=1),
        VAE_ResidualBlock(128, 128),
        VAE_ResidualBlock(128, 128),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
        VAE_ResidualBlock(128, 256),
        VAE_ResidualBlock(256, 256),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
        VAE_ResidualBlock(256, 512),
        VAE_ResidualBlock(512, 512),
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),
        VAE_AttentionBlock(512),
        VAE_ResidualBlock(512, 512),
        nn.GroupNorm(32, 512),
        nn.SiLU(),
        nn.Conv2d(512, 8, kernel_size=3, padding=1),
        nn.Conv2d(8, 8, kernel_size=1, padding=0)
    )

  def forward(self, x, noise):
    for module in self:
      if getattr(module, "stride", None) == (2,2):
        x = F.pad(x, (0, 1, 0, 1))
      x = module(x)

    mean, log_var = torch.chunk(x, 2, dim=1) # [b, 8,, h, w] - 2*[b, 4, h, w]
    log_var = torch.clamp(log_var, -30, 20)
    var = log_var.exp()
    stddev = var.sqrt()
    x = mean + stddev * noise
    x *= 0.18215 * x
    return x

