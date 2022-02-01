import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pt
import random
import yaml
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchmetrics 
from tqdm import tqdm
from UNetModel import UNet
from configurare_data import create_dataset_csv , split_dataset
from lungs_class import LungSegDataset , plot_acc_loss
from train_class import train


print(f"pyTorch version {torch.__version__}")
print(f"torchvision version {torchvision.__version__}")
print(f"torchmetrics version {torchmetrics.__version__}")
print(f"CUDA available {torch.cuda.is_available()}")

config = None
with open('config.yaml') as f: # reads .yml/.yaml files
    config = yaml.safe_load(f)
    

dataset_df = create_dataset_csv(config["data"]["images_dir"], 
                                config["data"]["right_masks_dir"],
                                config["data"]["left_masks_dir"],
                                config["data"]["data_csv"])

dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)
dataset_df.head(3)

data_ds = LungSegDataset(dataset_df, img_size=config["data"]["img_size"])
x, y = data_ds[0]
print(x.shape, y.shape)

f, axs = plt.subplots(1, 2)
axs[0].axis('off')
axs[0].set_title("Input")
axs[0].imshow(x[0].numpy(), cmap="gray")

axs[1].axis('off')
axs[1].set_title("Mask")
axs[1].imshow(y[0].numpy(), cmap="gray")

network = UNet(n_channels=1, n_classes=2)
print(network)

train_df = dataset_df.loc[dataset_df["subset"] == "train", :]
train_ds = LungSegDataset(train_df, img_size=config["data"]["img_size"])
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['train']['bs'], shuffle=True)

valid_df = dataset_df.loc[dataset_df["subset"] == "valid", :]
valid_ds = LungSegDataset(valid_df, img_size=config["data"]["img_size"])
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=config['train']['bs'], shuffle=False)

print(f"# Train: {len(train_ds)} # Valid: {len(valid_ds)}")

criterion = torch.nn.CrossEntropyLoss()

if config['train']['opt'] == 'Adam':
    opt = torch.optim.Adam(network.parameters(), lr=config['train']['lr'])
elif config['train']['opt'] == 'SGD':
    opt = torch.optim.SGD(network.parameters(), lr=config['train']['lr'])

history = train(network, train_loader, valid_loader, criterion, opt, epochs=config['train']['epochs'], thresh=config['test']['threshold'])

plot_acc_loss(history)