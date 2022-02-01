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
from lungs_class import LungSegDataset
import torch


def train(network, train_loader, valid_loader, criterion, opt, epochs, thresh=0.5):
    total_loss = {'train': [], 'valid': []}
    total_acc = {'train': [], 'valid': []}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting training on device {device} ...")

    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }
    metric = torchmetrics.Accuracy()
    
    network.to(device)
    criterion.to(device)

    for ep in range(epochs):
        print(f"[INFO] Epoch {ep}/{epochs - 1}")
        print("-" * 20)        
        for phase in ['train', 'valid']:
            running_loss = 0.0

            if phase == 'train':
                network.train()  # Set model to training mode
            else:
                network.eval()   # Set model to evaluate mode

            with tqdm(desc=phase, unit=' batch', total=len(loaders[phase].dataset)) as pbar:
                for data in loaders[phase]:
                    ins, tgs = data
                    ins = ins.to(device)
                    tgs = tgs.to(device)
                    
                    # seteaza toti gradientii la zero, deoarece PyTorch acumuleaza valorile lor dupa mai multe backward passes
                    opt.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # se face forward propagation -> se calculeaza predictia
                        output = network(ins)

                        # se calculeaza eroarea/loss-ul
                        loss = criterion(output, tgs.squeeze())
                        
                        # deoarece reteaua nu include un strat de softmax, predictia finala trebuie calculata manual
                        current_predict = (F.softmax(output, dim=1)[:, 1] > thresh).float()

                        if 'cuda' in device.type:
                            current_predict = current_predict.cpu()
                            current_target = tgs.cpu().type(torch.int).squeeze()
                        else:
                            current_predict = current_predict
                            current_target = tgs.type(torch.int).squeeze()

                        # print(current_predict.shape, current_target.shape)
                        # print(current_predict.dtype, current_target.dtype)
                        acc = metric(current_predict, current_target)
                        # print(f"\tAcc on batch {i}: {acc}")

                        if phase == 'train':
                            # se face backpropagation -> se calculeaza gradientii
                            loss.backward()
                            # se actualizează weights-urile
                            opt.step()
                    
                    running_loss += loss.item() * ins.size(0)
                    # print(running_loss, loss.item())

                    if phase == 'valid':
                        # salvam ponderile modelului dupa fiecare epoca
                        torch.save(network, 'my_model.pth')
                    
                    pbar.update(ins.shape[0])


                # Calculam loss-ul pt toate batch-urile dintr-o epoca
                total_loss[phase].append(running_loss/len(loaders[phase].dataset))
                
                # Calculam acuratetea pt toate batch-urile dintr-o epoca
                acc = metric.compute()
                total_acc[phase].append(acc)

                postfix = f'error {total_loss[phase][-1]:.4f} accuracy {acc*100:.2f}%'
                pbar.set_postfix_str(postfix)

                # Resetam pt a acumula valorile dintr-o noua epoca
                metric.reset()
    
    return {'loss': total_loss, 'acc': total_acc}