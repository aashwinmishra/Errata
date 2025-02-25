import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import os
import argparse
from data_setup import unzip_data, get_dataloaders
from utils import get_devices, set_seeds, show_batch
from models import AnimeDiscriminator, AnimeGenerator
from engine import train


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./drive/MyDrive/Practice_Data/AnimeFacesDatasetKaggle.zip")
parser.add_argument("--data_dir", type=str, default="./Data/")
parser.add_argument("--num_epochs", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("--image_size", type=int, default=64)
args = parser.parse_args()

unzip_data(args.data_path, args.data_dir)
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(*stats)])
dl = get_dataloaders(args.data_dir, transform, args.batch_size)
device = get_devices()
disc = AnimeDiscriminator().to(device)
gen = AnimeGenerator().to(device)
train(gen, disc, dl, args.batch_size, args.latent_dim, args.num_epochs, "./generated", device)
