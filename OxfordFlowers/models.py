import torch
import torch.nn as nn


def ConvBlock(in_channels: int, out_channels: int):
  return nn.Sequential(
      nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(in_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2,2)
  )


class MyModel(nn.Module):
  def __init__(self, 
               input_dim: int, 
               in_channels: int, 
               num_classes: int):
    super().__init__()
    self.block1 = ConvBlock(in_channels, 64)  #[3, 224, 224] -> [64, 112, 112] s/2
    self.block2 = ConvBlock(64, 128)          #[64, 112, 112]-> [128, 56, 56]  s/4
    self.block3 = ConvBlock(128, 256)         #[128, 56, 56] -> [256, 28, 28]  s/8
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.1),
        nn.Linear(256 * int(input_dim**2/64), 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
  
  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    return self.classifier(x)

