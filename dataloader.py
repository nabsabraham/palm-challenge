#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 22:03:11 2019

@author: nabila
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
import PIL.ImageOps
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, ConcatDataset
from torchvision import transforms

class palm(Dataset): 
    def __init__(self, path, img_list, transform=None, train=True):
        self.path = path
        self.transform = transform           
        self.img_list = img_list
        self.train = train
        
    def __getitem__(self, index):
        img_name = self.img_list[index]
        
        if self.train: 
            img = Image.open(os.path.join(self.path, "train", img_name))
            mask_name = img_name.strip('.jpg') + '.bmp'
            mask = Image.open(os.path.join(self.path, "masks", mask_name ))
            mask = PIL.ImageOps.invert(mask)

            if self.transform is not None: 
                img = self.transform(img)
                mask = self.transform(mask)
                return img, mask
    
        else:
            img = Image.open(os.path.join(self.path, "val", img_name))
            if self.transform is not None: 
                img = self.transform(img)
                return img
            
    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":    
    
    root_dir = "/home/nabila/Desktop/datasets/PALM"
    img_list = os.listdir(os.path.join(root_dir, 'train'))
    gt_list = os.listdir(os.path.join(root_dir, 'masks'))
    test_list = os.listdir(os.path.join(root_dir, 'val'))
    
    batch_size = 8
    val_split = 0.1
    imsize = 224
    num_train = len(img_list)
    indices = np.arange(num_train)
    split = int(np.floor(val_split*num_train))
    train_idx, val_idx = indices[split:], indices[:split]
    
    
    transformations = transforms.Compose([transforms.Resize((imsize, imsize)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0,0,0],
                                                               [1,1,1])])
    
    train_data = palm(root_dir, np.array(img_list)[train_idx], 
                      transform=transformations, train=True)
    val_data = palm(root_dir, np.array(img_list)[val_idx], 
                    transform=transformations, train=True)
    test_data = palm(root_dir, np.array(test_list), 
                    transform=transformations, train=False)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    #visualize train datasets
    X,Y = next(iter(train_loader))
    plt.figure()
    plt.title('Images')
    grid_img = vutils.make_grid(X, nrow=4)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.figure()
    plt.title('Ground Truths')
    gt_grid = vutils.make_grid(Y, nrow=4)
    plt.imshow(gt_grid.permute(1,2,0))
    
    # mean and std of training data (90% of train data size)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for X, _ in train_loader:
        batch_samples = X.size(0)
        X = X.view(batch_samples, X.size(1), -1)
        mean += X.mean(2).sum(0)
        std += X.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    print("mean of dataset:", mean)
    print("std of dataset:", std)


    # visualize val data 
    X, Y = next(iter(val_loader))
    plt.figure()
    plt.title('Val Images')
    grid_img = vutils.make_grid(X, nrow=4)
    plt.imshow(grid_img.permute(1, 2, 0))

    plt.figure()
    plt.title('Val Ground Truths')
    gt_grid = vutils.make_grid(Y, nrow=4)
    plt.imshow(gt_grid.permute(1,2,0))

    # visualize test data
    X = next(iter(test_loader))
    plt.figure()
    plt.title('Test Images')
    grid_img = vutils.make_grid(X, nrow=4)
    plt.imshow(grid_img.permute(1, 2, 0))

