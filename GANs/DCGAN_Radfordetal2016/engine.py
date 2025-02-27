import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt



def discriminator_step(discriminator, 
                       opt_d, 
                       generator, 
                       real_images, 
                       batch_size, 
                       latent_dim, 
                       device):
  real_preds = discriminator(real_images)
  real_targets = torch.ones_like(real_preds, device=device)
  real_loss = torch.nn.functional.binary_cross_entropy(real_preds, real_targets)

  latents = torch.randn(size=(batch_size, latent_dim, 1, 1), device=device)
  fake_images = generator(latents)
  fake_preds = discriminator(fake_images)
  fake_targets = torch.zeros_like(fake_preds, device=device)
  fake_loss = torch.nn.functional.binary_cross_entropy(fake_preds, fake_targets)

  opt_d.zero_grad()
  loss = real_loss + fake_loss 
  loss.backward()
  opt_d.step()


def generator_step(generator, 
                   opt_g, 
                   discriminator, 
                   batch_size, 
                   latent_dim, 
                   device):
  latents = torch.randn(size=(batch_size, latent_dim, 1, 1), device=device)
  fake_images = generator(latents)
  fake_preds = discriminator(fake_images)
  fake_targets = torch.ones_like(fake_preds, device=device)
  loss = torch.nn.functional.binary_cross_entropy(fake_preds, fake_targets)

  opt_g.zero_grad()
  loss.backward()
  opt_g.step()


def train(discriminator, 
          generator, 
          train_dl, 
          batch_size, 
          latent_dim, 
          num_epochs, 
          device):
  opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
  opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
  fixed_latents = torch.randn(size=(64, latent_dim, 1, 1), device=device)

  with torch.inference_mode():
    fake_images = generator(fixed_latents)
    fake_images = torch.reshape(fake_images, (64, 1, 64, 64))
    image = (torchvision.utils.make_grid(fake_images.detach().cpu()[:64], nrow=8) + 1.0)/2.0
    plt.figure(figsize=(8,8))
    plt.imshow(image.permute(1,2,0))
    plt.axis('off')
    plt.savefig("0.png", bbox_inches='tight')

  for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1} of {num_epochs}...")
    for xb, yb in train_dl:
      xb = xb.to(device)
      discriminator_step(discriminator, opt_d, generator, xb, batch_size, latent_dim, device)
      generator_step(generator, opt_g, discriminator, batch_size, latent_dim, device)

    with torch.inference_mode():
      fake_images = generator(fixed_latents)
      fake_images = torch.reshape(fake_images, (64, 1, 64, 64))
      image = (torchvision.utils.make_grid(fake_images.detach().cpu()[:64], nrow=8) + 1.0)/2.0
      plt.figure(figsize=(8,8))
      plt.imshow(image.permute(1,2,0))
      plt.axis('off')
      plt.savefig(f"{epoch + 1}.png", bbox_inches='tight')
