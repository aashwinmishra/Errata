"""
Utilities to train and evaluate speified model on speified data.
"""
import torch
import torchvision
import argparse
import glob
import os

from data_setup import get_data, get_dataloader
from utils import get_devices, set_seeds, accuracy, save_model
from models_additional import ViT_Base
from engine import train

parser = argparse.ArgumentParser()
parser.add_argument('--url', type=str, default="https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip")
parser.add_argument('--project_name', type=str, default="ViT_Base")
parser.add_argument('--model_name', type=str, default="ViT_Base")
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

get_data(args.url, "./Data", args.project_name)

path = "./Data" + "/" + args.project_name + "/train/"
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((args.img_size, args.img_size)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
train_dl = get_dataloader(path, train_transforms, args.batch_size, True)

val_path = "./Data" + "/" + args.project_name + "/test/"
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((args.img_size, args.img_size)),
    torchvision.transforms.ToTensor()
])
val_dl = get_dataloader(val_path, val_transforms, args.batch_size, False)

set_seeds()
device = get_devices()
model = ViT_Base(in_channels=3, 
                 img_size=args.img_size, 
                 patch_size=16, 
                 num_layers=12, 
                 num_heads=12, 
                 embed_dim=768, 
                 mlp_size=3072, 
                 mlp_dropout=0.1, 
                 num_classes=3).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = torch.nn.CrossEntropyLoss()
metric_fn = accuracy
history = train(model, train_dl, val_dl, loss_fn, opt, metric_fn, device, args.num_epochs)
save_model(model, "./Models", args.model_name)
