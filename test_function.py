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
from UNet import UNet

def test(network, test_loader, thresh=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting testing on device {device} ...")

    metric = torchmetrics.Accuracy()

    network.eval()
    with tqdm(desc='test', unit=' batch', total=len(test_loader.dataset)) as pbar:
        for data in test_loader:
            ins, tgs = data
            ins = ins.to(device)
            tgs = tgs.to('cpu')

            output = network(ins)
            current_predict = (F.softmax(output, dim=1)[:, 1] > thresh).float()

            if 'cuda' in device.type:
                current_predict = current_predict.cpu()
                
            acc = metric(current_predict, tgs.squeeze())
            pbar.update(ins.shape[0])
        
        acc = metric.compute()
        print(f'[INFO] Test accuracy is {acc*100:.2f} %')