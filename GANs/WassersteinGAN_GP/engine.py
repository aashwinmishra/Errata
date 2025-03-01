import torch
import torch.nn as nn
import torchvision
from utils import gradient_penalty
import matplotlib.pyplot as plt


def critic_step(critic,
                opt_c,
                generator,
                real_images,
                batch_size,
                latent_dim,
                lambda_gp,
                device):
  real_preds = critic(real_images)
  latents = torch.randn(size=(batch_size, latent_dim, 1, 1), device=device)
  fake_images = generator(latents)
  fake_preds = critic(fake_images)

  gp = gradient_penalty(critic, real_images, fake_images, device)

  opt_c.zero_grad()
  loss = -real_preds.mean() + fake_preds.mean() + lambda_gp * gp
  loss.backward()
  opt_c.step()

  # for param in critic.parameters():
  #   param.data.clamp_(-weight_clip, weight_clip)


def generator_step(generator,
                   opt_g,
                   critic,
                   batch_size,
                   latent_dim,
                   device):
  latents = torch.randn(size=(batch_size, latent_dim, 1, 1), device=device)
  fake_images = generator(latents)
  loss = -critic(fake_images).mean()

  opt_g.zero_grad()
  loss.backward()
  opt_g.step()


def train(critic,
          generator,
          train_dl,
          batch_size,
          latent_dim,
          num_epochs,
          critic_iterations,
          lambda_gp,
          device):
  opt_c = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
  opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
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
    for step, (xb, yb) in enumerate(train_dl):
      xb = xb.to(device)
      critic_step(critic, opt_c, generator, xb, batch_size, latent_dim, lambda_gp, device)
      if (step % critic_iterations) == 0:
        generator_step(generator, opt_g, critic, batch_size, latent_dim, device)

    with torch.inference_mode():
      fake_images = generator(fixed_latents)
      fake_images = torch.reshape(fake_images, (64, 1, 64, 64))
      image = (torchvision.utils.make_grid(fake_images.detach().cpu()[:64], nrow=8) + 1.0)/2.0
      plt.figure(figsize=(8,8))
      plt.imshow(image.permute(1,2,0))
      plt.axis('off')
      plt.savefig(f"{epoch + 1}.png", bbox_inches='tight')

