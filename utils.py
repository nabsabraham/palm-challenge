#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:07:15 2019

@author: nabila
"""

import matplotlib.pyplot as plt 
import numpy as np
import torch.nn as nn

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()



def plot_hist(epochs, train, val, data):
    plt.figure(dpi=150)
    e = np.arange(1, epochs+1,1)
    plt.plot(e, train, label='Train'+ " " + str(data))
    plt.plot(e, val, label='Val'+ " " + str(data))
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(str(data) + " Values")
    plt.grid()
    plt.title(str(data) + " Results")
    
def dice_numpy(pred, gt):
    eps = 1e-7
    intersection = (pred*gt)
    dsc = (intersection.sum().sum() + eps) / (pred.sum().sum() + gt.sum().sum() + eps)
    return dsc
