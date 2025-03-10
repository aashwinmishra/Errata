import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.linear_1 = nn.Linear(n_embd, 4*n_embd)
    self.linear_2 = nn.Linear(4*n_embd, 4*n_embd)

  def forward(self, x):
    x = self.linear_1(x)
    x = F.silu(x)
    return self.linear_2(x)


class Diffusion(nn.Module):
  def __init__(self):
    super().__init__()
    self.time_embedding = TimeEmbedding(320)
    self.unet = UNET()
    self.final = UNET_OutputLayer(320)

  def forward(self, latent, context, time):
    #latent: [B, 4, h, w]
    #context: [B, S, 768]
    #time: [1, 320]
    time = self.time_embedding(time)
    output = self.unet(latent)
    output = self.final(output)
    return output 

