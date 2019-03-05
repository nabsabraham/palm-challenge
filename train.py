#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:38:19 2019

@author: nabila
"""

import os
import torch
import numpy as np
from PIL import Image 
from glob import glob
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms, models
from torchsummary import summary 

from dataloader import palmFromText
import utils 
import losses
import unet


data_path = '/home/nabila/Desktop/datasets/PALM'

imsize=1024
epochs = 25
batch_size = 1

t = transforms.Compose([transforms.Resize((imsize,imsize)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0,0,0],
                                                           [1,1,1])])

train_data = palmFromText(data_path, "train.txt", transform=t)
val_data = palmFromText(data_path, "trainval.txt", transform=t)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

#X,Y = next(iter(train_loader))
#plt.figure()
#plt.title('Images')
#grid_img = vutils.make_grid(X, nrow=4)
#plt.imshow(grid_img.permute(1, 2, 0))
#plt.figure()
#plt.title('Ground Truths')
#gt_grid = vutils.make_grid(Y, nrow=4)
#plt.imshow(gt_grid.permute(1,2,0))


model = unet.small_unet(n_channels=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#summary(model, input_size=(3, 144, 144))

opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
print('='*30)
print('Training')
print('='*30)

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

for e in range(epochs):
    train_loss = 0
    train_acc = 0
    steps = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        opt.zero_grad()
        preds = model(images)

        loss = losses.dice_loss(preds, masks)
        
        loss.backward()
        opt.step()
        train_loss += loss.detach().item()
        train_acc += losses.dice_score(preds, masks).detach().item()

    else:        
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                preds = model.forward(inputs)
                batch_loss = losses.dice_loss(preds, masks)
                
                val_loss += batch_loss.detach().item()
                val_acc += losses.dice_score(preds, masks).detach().item()
                
        print("Epoch {}/{} \t Train loss:{:.5} \t Train Acc:{:.5} \t Val Loss:{:.5} \t Val Acc:{:.5}".format(
                e+1, epochs, 
                train_loss/len(train_loader), 
                train_acc/len(train_loader),
                val_loss/len(val_loader), 
                val_acc/len(val_loader)))
        
        train_loss_history.append(train_loss/len(train_loader))
        val_loss_history.append(val_loss/len(train_loader))
        train_acc_history.append(train_acc/len(train_loader))
        val_acc_history.append(val_acc/len(train_loader))

        model.train()

utils.plot_hist(epochs, train_loss_history, val_loss_history, 'Loss')
utils.plot_hist(epochs, train_acc_history, val_acc_history, 'Accuracy')

# check model outputs on validation data
model.eval()
idx = np.random.randint(0,batch_size)

val_dsc = []
with torch.no_grad():
    for x_val, y_val in val_loader:
        x_val, y_val = x_val.to(device), y_val.to(device)
        preds = model(x_val)
        dsc = losses.dice_score(preds, y_val)
        val_dsc.append(dsc/x_val.shape[0])
    

print('='*30)
print('Average DSC score =', np.array(val_dsc).mean())

grid_img = vutils.make_grid(y_val)
grid_pred = vutils.make_grid(preds)
plt.subplot(121)
plt.imshow(grid_pred.permute(1,2,0))
plt.title("predictions")

plt.subplot(122)
plt.imshow(grid_img.permute(1,2,0))
plt.title("gt")

