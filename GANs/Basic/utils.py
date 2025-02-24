import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def get_devices() -> torch.device:
  """
  Returns the appropriate device
  """
  return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seeds(seed: int=42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)


def denorm(images, stats=(0.5, 0.5)):
  return images * stats[0] + stats[1]


def show_batch(images, nrow=8, save=False, save_name: str=None):
  image = torchvision.utils.make_grid(denorm(images.detach())[:64], nrow=nrow)
  plt.figure(figsize=(8,8))
  plt.imshow(image.permute(1,2,0))
  plt.axis('off')
  if save:
    plt.savefig(save_name, bbox_inches='tight')

