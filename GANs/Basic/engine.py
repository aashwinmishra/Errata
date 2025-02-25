import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import show_batch
import os


def discriminator_step(discriminator: nn.Module, 
                       opt_d: torch.optim.Optimizer, 
                       real_images: torch.tensor, 
                       generator: nn.Module, 
                       latent_dim: int, 
                       device: torch.device):
  batch_size = real_images.shape[0]

  real_preds = discriminator(real_images)
  real_targets = torch.ones_like(real_preds, device=device)
  real_loss = F.binary_cross_entropy(real_preds, real_targets)

  latents = torch.randn(size=(batch_size, latent_dim, 1, 1), device=device)
  fake_images = generator(latents)
  fake_preds = discriminator(fake_images)
  fake_targets = torch.zeros_like(fake_preds, device=device)
  fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)

  opt_d.zero_grad()
  loss = real_loss + fake_loss
  loss.backward()
  opt_d.step()


def generator_step(generator: nn.Module, 
                   opt_g: torch.optim.Optimizer, 
                   discriminator: nn.Module, 
                   batch_size: int, 
                   latent_dim: int, 
                   device: torch.device):
  latents = torch.randn(size=(batch_size, latent_dim, 1, 1), device=device)
  fake_images = generator(latents)
  fake_preds = discriminator(fake_images)
  fake_targets = torch.ones_like(fake_preds, device=device)
  loss = F.binary_cross_entropy(fake_preds, fake_targets)

  opt_g.zero_grad()
  loss.backward()
  opt_g.step()


def train(generator: nn.Module, 
          discriminator: nn.Module, 
          dl: torch.utils.data.DataLoader,
          batch_size: int, 
          latent_dim: int, 
          num_epochs: int, 
          save_dir: str,
          device: torch.device):
  opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
  opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

  os.makedirs(save_dir, exist_ok=True)
  fixed_latents = torch.randn(size=(batch_size, latent_dim, 1, 1), device=device)
  with torch.inference_mode():
    fixed_images = generator(fixed_latents)
    temp_name = f"generated_{0}"
    show_batch(fixed_images.cpu().detach(), nrow=8, save=True, save_name=save_dir+"/"+temp_name)
  
  for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch + 1}")
    for xb, yb in dl:
      xb = xb.to(device)
      discriminator_step(discriminator, opt_d, xb, generator, latent_dim, device)
      generator_step(generator, opt_g, discriminator, batch_size, latent_dim, device)

    with torch.inference_mode():
      fixed_images = generator(fixed_latents)
      temp_name = f"generated_{epoch+1}"
      show_batch(fixed_images.cpu().detach(), nrow=8, save=True, save_name=save_dir+"/"+temp_name)

  
