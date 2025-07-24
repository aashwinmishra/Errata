"""
Utilities to download and process data.
"""
import torch
import torchvision
from PIL import Image
import os
import glob
import requests
import zipfile


def get_data(url: str, data_dir: str, proj_name: str) -> None:
  """
  Downloads, unzips and saves dataset at specified location.

  Args:
    url: Web address to access data.
    data_dir: Directory to save data.
    proj_name: Project/subfolder name.
  """
  path = data_dir + "/" + proj_name
  if not os.path.isdir(path):
    os.makedirs(path, exist_ok=True)
    with open(path + ".zip", "wb") as f:
      request = requests.get(url)
      f.write(request.content)

      with zipfile.ZipFile(path + ".zip", "r") as zip_ref:
        zip_ref.extractall(path)
      os.remove(path + ".zip")


class CustomImageFolder(torch.utils.data.Dataset):
  """
  Custom implimentation similar to torchvision's ImageFolder class.

  Attributes:
    classes: a sorted list of the names of the different classes in the dataset.
    class_to_idx: a dictionary mapping the class names to their categorical representation.
    image_paths: a list of absolute paths to all images in the dataset.
    transforms: torchvision transforms to be applied to the individual images.
  """

  def __init__(self, data_dir: str,
               transform: torchvision.transforms.Compose):
    """Initializes a new CustomImageFolder object.

    Args:
      data_dir: path to folder where data is stored.
      transforms: torchvision transforms to be applied to individual images.
    """
    self.classes = sorted(next(os.walk(data_dir))[1])
    self.class_to_idx = {c:i for i, c in enumerate(self.classes)}
    self.image_paths = [entry for entry in glob.glob(data_dir+"**/*", recursive=True) if os.path.isfile(entry)]
    self.transform = transform

  def __len__(self) -> int:
    return len(self.image_paths)

  def __getitem__(self, index: int):
    """Retrieves the (image, label) tuple from the dataset list by index.

    Args:
      index: Index of image in the dataset image_paths list.

    Returns:
      (image, label) tuple corresponding to the index.

    Raises:
      IndexError: if the index is out of range.
    """
    path = self.image_paths[index]
    img_arr = Image.open(path)
    label = self.class_to_idx[path.split("/")[-2]]
    if self.transform:
      img_arr = self.transform(img_arr)
    return img_arr, label


def get_dataloader(path: str,
                   transform: torchvision.transforms.Compose,
                   batch_size: int,
                   shuffle: bool) -> torch.utils.data.DataLoader:
  """Creates a Dataset & DataLoader for the data at path.

  Args:
    path: Absolute path of the data.
    transform: Torchvision transforms to be applied in Dataset.
    batch_size: Batch size in the DataLoader
    shuffle: If to shuffle data in the DataLoader

  Returns:
    DataLoader corresponding to the data.
  """
  ds = CustomImageFolder(data_dir=path, transform=transform)
  return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
