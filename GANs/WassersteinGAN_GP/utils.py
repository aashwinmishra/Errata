import torch
import torch.nn as nn


def get_devices() -> torch.device:
  """
  Returns gpu device if available, else cpu
  """
  return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seeds(seed: int = 42):
  """
  Sets torch seeds to ensure reproducability.
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)


def gradient_penalty(critic, real_imgs, fake_imgs, device):
  batch_size, c, h, w = real_imgs.shape
  epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
  interpolated_images = real_imgs * epsilon + fake_imgs * (1.0 - epsilon)

  scores = critic(interpolated_images)
  gradient = torch.autograd.grad(
      inputs=interpolated_images,
      outputs=scores,
      grad_outputs=torch.ones_like(scores),
      create_graph=True,
      retain_graph=True
  )[0]
  gradient = gradient.view(gradient.shape[0], -1)
  gradient_norm = gradient.norm(2, dim=-1)
  gradient_penalty = torch.mean((gradient_norm -1)**2)
  return gradient_penalty

