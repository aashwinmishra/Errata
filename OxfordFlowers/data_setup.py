import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import requests 
import tarfile
import os
import scipy.io
from PIL import Image


def get_data(images_url: str, 
             labels_url: str, 
             image_file: str="images.tgz", 
             labels_file: str="labels.mat",
             data_dir: str="./Data"):
  os.makedirs(data_dir, exist_ok=True)
  r = requests.get(images_url, allow_redirects=True)
  open(image_file, 'wb').write(r.content)
  with tarfile.open(image_file, "r:gz") as tar:
    tar.extractall(path=data_dir)
  r = requests.get(labels_url, allow_redirects=True)
  open(labels_file, 'wb').write(r.content)


class FlowersDataset(Dataset):
  def __init__(self, 
               image_dir: str="./Data/jpg/", 
               labels_file: str="./labels.mat", 
               transform: callable=transforms.ToTensor()):
    self.image_paths = []
    for root, dirs, files in os.walk(image_dir):
      for file in files:
        full_path = os.path.join(root, file)
        self.image_paths.append(full_path)
    self.labels = (scipy.io.loadmat(labels_file)['labels'].squeeze()-1).tolist()
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    img = Image.open(self.image_paths[index])
    label = self.labels[index] 
    img = self.transform(img)
    return img, label


class TransformedSubset(Dataset):
  def __init__(self, subset, transform=None):
    self.subset = subset
    self.transform = transform
        
  def __getitem__(self, index):
    x, y = self.subset[index]
    if self.transform:
      x = self.transform(x)
    return x, y
        
  def __len__(self):
    return len(self.subset)
    
