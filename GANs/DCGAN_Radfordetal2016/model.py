import torch
import torch.nn as nn


def ConvBlock(in_channels, out_channels, kernel_size, stride, padding):
  return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2)
  )


def ConvTransposeBlock(in_channels, out_channels, kernel_size, stride, padding):
  return nn.Sequential(
      nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
  )


class Discriminator(nn.Module):
  def __init__(self, 
               channels_img, 
               features_d):
    super().__init__()
    self.disc = nn.Sequential(
        nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        ConvBlock(features_d, 2*features_d, 4, 2, 1),
        ConvBlock(2*features_d, 4*features_d, 4, 2, 1),
        ConvBlock(4*features_d, 8*features_d, 4, 2, 1),
        nn.Conv2d(8*features_d, 1, 4, 2, 0),
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.disc(x)


class Generator(nn.Module):
  def __init__(self, latent_dim, channels_img, features_g):
    super().__init__()
    self.gen = nn.Sequential(
        ConvTransposeBlock(latent_dim, features_g*16, 4, 1, 0),
        ConvTransposeBlock(features_g*16, features_g*8, 4, 2, 1),
        ConvTransposeBlock(features_g*8, features_g*4, 4, 2, 1),
        ConvTransposeBlock(features_g*4, features_g*2, 4, 2, 1),
        nn.ConvTranspose2d(features_g*2, channels_img, 4, 2, 1),
        nn.Tanh()
    )

  def forward(self, x):
    return self.gen(x)


def initialize_weights(model):
  for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
      nn.init.normal_(module.weight.data, 0.0, 0.02)

