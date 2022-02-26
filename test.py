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
from lungs_class import LungSegDataset
from configurare_data import create_dataset_csv , split_dataset
from test_function import test
from datetime import datetime
import os 

from lungs_class import plot_acc_loss
config = None
with open('config.yaml') as f: # reads .yml/.yaml files
    config = yaml.safe_load(f)
    
yml_data=yaml.dump(config)
directory =f"Test{datetime.now().strftime('%m%d%Y_%H%M')}"
parent_dir =r'D:\ai intro\Pytorch\Segmentare_pytorch\Experiment_Dice_index02222022_1339'
path = os.path.join(parent_dir, directory)
os.mkdir(path)

f= open(f"{path}\\yaml_config.txt","w+")
f.write(yml_data)    

dataset_df = create_dataset_csv(config["data"]["images_dir"], 
                                config["data"]["right_masks_dir"],
                                config["data"]["left_masks_dir"],
                                config["data"]["data_csv"])

dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)
dataset_df.head(3)

test_df = dataset_df.loc[dataset_df["subset"] == "test", :]
test_ds = LungSegDataset(test_df, img_size=config["data"]["img_size"])
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["train"]["bs"], shuffle=False)

network = torch.load(r"D:\ai intro\Pytorch\Segmentare_pytorch\Experiment_Dice_index02222022_1339\Weights\my_model02222022_1434_e95.pt")

print(f"# Test: {len(test_ds)}")


test(network, test_loader, thresh=config['test']['threshold'])

x, y = next(iter(test_loader))
network.eval()
y_pred = network(x.to(device='cuda'))
y_pred.shape

nr_exs = 4 # nr de exemple de afisat
fig, axs = plt.subplots(nr_exs, 3, figsize=(10, 10))
for i, (img, gt, pred) in enumerate(zip(x[:nr_exs], y[:nr_exs], y_pred[:nr_exs])):
    axs[i][0].axis('off')
    axs[i][0].set_title('Input')
    axs[i][0].imshow(img[0], cmap='gray')

    axs[i][1].axis('off')
    axs[i][1].set_title('Ground truth')
    axs[i][1].imshow(gt[0], cmap='gray')

    # print(pred.shape)
    pred = F.softmax(pred, dim=0)[1].detach().cpu().numpy()
    # print(pred.shape, pred.min(), pred.max())
    pred[pred > config['test']['threshold']] = 1
    pred[pred <= config['test']['threshold']] = 0
    pred = pred.astype(np.uint8)

    axs[i][2].axis('off')
    axs[i][2].set_title('Prediction')
    axs[i][2].imshow(pred, cmap='gray')
    
plt.savefig(f"{path}\\Măști")     


       
        