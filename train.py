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
import models

data_path = '/home/nabila/Desktop/datasets/PALM'

imsize=1440
num_epochs = 50
batch_size = 1

t = transforms.Compose([transforms.Resize((imsize,imsize)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0,0,0],
                                                           [1,1,1])])

train_data = palmFromText(data_path, "txtfiles/train.txt", transform=t)
val_data = palmFromText(data_path, "txtfiles/trainval.txt", transform=t)

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


model = models.unet(n_channels=3, n_classes=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#summary(model, input_size=(3, 144, 144))

opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, verbose=True)
early_stopping = utils.EarlyStopping(patience=8, verbose=True)

print('='*30)
print('Training')
print('='*30)
epoch_train_loss = []
epoch_val_loss = []
epoch_train_dsc = []
epoch_val_dsc = []


for epoch in range(num_epochs):
    train_losses = []
    train_dsc = []
    val_losses = []
    val_dsc = []

    steps = 0
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        model.train()
        opt.zero_grad()
        preds = model(images)

        loss = losses.dice_loss(preds, masks)
        
        loss.backward()
        #opt.step()
        
        train_losses.append(loss.item())
        train_dsc.append(losses.dice_score(preds, masks).item())


    else:        
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                preds = model.forward(inputs)
                loss = losses.dice_loss(preds, masks)

                val_losses.append(loss.item())
                val_dsc.append(losses.dice_score(preds,masks).item())
                scheduler.step(loss)
                
    print('[%d]/[%d] Train Loss:%.4f\t Train Acc:%.4f\t Val Loss:%.4f\t Val Acc: %.4f'
            % (epoch+1, num_epochs, 
               np.mean(train_losses),  np.mean(train_dsc),  
               np.mean(val_losses),  np.mean(val_dsc)))
    
    epoch_train_loss.append(np.mean(train_losses))
    epoch_val_loss.append(np.mean(val_losses))
    epoch_train_dsc.append(np.mean(train_dsc))
    epoch_val_dsc.append(np.mean(val_dsc))
    
    early_stopping(np.average(val_losses), model)
    
    if early_stopping.early_stop:
        print("Early stopping at epoch: ", epoch)
        break

print('='*30)
print('Average DSC score =', np.array(val_dsc).mean())


utils.plot_hist(epoch, np.array(epoch_train_loss), np.array(epoch_val_loss), "Loss")
utils.plot_hist(epoch, np.array(epoch_train_dsc), np.array(epoch_val_dsc), "DSC Score")



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

