
import torch
import pytorch_lightning as L
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class MnistDataModule(L.LightningDataModule):
  def __init__(self, 
               data_dir: str="./", 
               transforms: transforms.transforms=transforms.ToTensor(),
               batch_size: int=32):
    super().__init__()
    self.data_dir = data_dir
    self.transform = transforms 
    self.batch_size = batch_size

  def prepare_data(self):
    datasets.MNIST(self.data_dir, train=True, download=True)
    datasets.MNIST(self.data_dir, train=False, download=True)

  def setup(self, stage: str):
    if stage == 'fit':
      mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
      self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_full, [0.5, 0.5], generator=torch.Generator().manual_seed(42))

    if stage == 'test':
      self.minst_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    if stage == 'predict':
      self.minst_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size)

  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size)

  def test_dataloader(self):
    return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size)

  def predict_dataloader(self):
    return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size)

