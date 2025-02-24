import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from zipfile import ZipFile
import os


def unzip_data(data_path: str, data_dir: str) -> None:
  """
  Takes in data zip file path, unzips and stores to data_dir.
  Args:
    data_path: Path to the zip file with data
    data_dir: Directory to unzip and store data at.
  """
  os.makedirs(data_dir, exist_ok=True)
  with ZipFile(data_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)


def get_dataloaders(data_dir: str,
                    transforms: torchvision.transforms,
                    batch_size: int=128) -> torch.utils.data.DataLoader:
  """
  Takes data path and returns torch dataloader instance.
  Args:
    data_dir: directory where data is located
    transforms: Torchvision transforms to transform data
    batch_size: Size of batches in dataloader.
  Returns:
    DataLoader instance
  """
  return DataLoader(torchvision.datasets.ImageFolder(data_dir, transform=transforms),
                  batch_size=batch_size, shuffle=True)

