import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision 
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import Generator, Discriminator, initialize_weights
from engine import train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
num_epochs = 5 
features_d = 64
features_g = 64
channels_img = 1
batch_size = 128
image_size = 64

transform = transforms.Compose([transforms.Resize(image_size), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,),(0.5,))
                                ])
dataset = datasets.MNIST(root = 'data/', download = True, transform=transform)
train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
gen = Generator(latent_dim, channels_img, features_g).to(device)
disc = Discriminator(channels_img, features_d).to(device)
initialize_weights(gen)
initialize_weights(disc)

train(disc, gen, train_dl, batch_size, latent_dim, num_epochs, device)

