"""
Utility functions to get device, set seeds, save models, etc.
"""
import torch
import numpy as np
import os


def get_devices() -> torch.device:
  """
  Returns cuda device if available, else cpu device.
  """
  return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seeds(seed: int=42) -> None:
  """
  Sets torch and numpy seeds for rng.

  Args:
    seed: Integer to initialize the pseudorandom number generator.
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)


def save_model(model: torch.nn.Module,
               save_dir: str,
               model_name: str) -> None:
  """
  Saves the PyTorch model at location save_dir.

  Args:
    model: model to be saved
    save_dir: Absolute path of the directory to save model to.
    model_name: name of the model to be saved as.
  """
  os.makedirs(save_dir, exist_ok=True)
  if not model_name.endswith("pt") and not model_name.endswith("pth"):
    model_name += ".pt"
  path = save_dir + "/" + model_name
  torch.save(model.state_dict(), path)


def accuracy(outputs: torch.tensor, labels: torch.tensor) -> float:
  """Computes accuracy over a batch given logits and labels.

  Args:
    outputs: Logit scores outputted by the model for the batch.
    labels: true labels for the batch.

  Returns:
    accuracy scores as a pure python float value in [0, 1]
  """
  with torch.inference_mode():
    predictions = torch.argmax(outputs, dim=-1)
    return (((predictions == labels).sum()).float() / len(labels)).item()

