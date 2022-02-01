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


class LungSegDataset(torch.utils.data.Dataset):
    """Un Dataset custom pentru setul de date pentru segmentare plamanilor"""

    def __init__(self, dataset_df, img_size):
        self.dataset_df = dataset_df.reset_index(drop=True)
        self.img_size = tuple(img_size)

    def __len__(self):
        """
        Returns:
            int: Returneaza numarul total de samples
        """
        return len(self.dataset_df) 

    def __combine_masks(self, img_right, img_left):
        """Combina mastile pentru cei doi plamani intr-o singura masca

        Args:
            img_right (pillow.Image): masca pentru plamanul drept
            img_left (pillow.Image): masca pentru plamanul stang

        Returns:
            numpy.array: masca cu cei doi plamani
        """

        img_right = np.array(img_right, dtype="uint8") * 1/255
        img_left = np.array(img_left, dtype="uint8") * 1/255

        img = (img_right + img_left).astype("uint8")

        return img


    def __getitem__(self, idx):
        """Returneaza un tuple (input, target) care corespunde cu batch #idx.

        Args:
            idx (int): indexul batch-ului curent

        Returns:
           tuple:  (input, target) care corespunde cu batch #idx
        """

        row = self.dataset_df.iloc[idx, :]

        img = cv2.imread(str(row['image_path']), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)

        # adaugam o noua dimensiune pe prima pozitie
        # o retea in pytorch asteapta input de forma C x H x W
        x = np.expand_dims(img, axis=0)

        mask_right = cv2.imread(str(row['right_lung_mask_path']), cv2.IMREAD_GRAYSCALE)
        mask_right = cv2.resize(mask_right, self.img_size)
        mask_left = cv2.imread(str(row['left_lung_mask_path']), cv2.IMREAD_GRAYSCALE)
        mask_left = cv2.resize(mask_left, self.img_size)

        mask = self.__combine_masks(mask_right, mask_left)

        y = np.expand_dims(mask, axis=0)
        
        return torch.as_tensor(x.copy()).float(), torch.as_tensor(y.copy()).long()
    
def plot_acc_loss(result):
    acc = result['acc']['train']
    loss = result['loss']['train']
    val_acc = result['acc']['valid']
    val_loss = result['loss']['valid']
    
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('Accuracy', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    
    plt.subplot(122)
    plt.plot(loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.show()
    
