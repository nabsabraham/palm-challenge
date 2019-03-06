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

# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({.6f} --> {.6f}).  Saving model ...'.format(self.val_loss_min,val_loss ))
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# https://github.com/pytorch/examples/blob/537f6971872b839b36983ff40dafe688276fe6c3/imagenet/main.py#L340
# https://github.com/pytorch/examples/blob/537f6971872b839b36983ff40dafe688276fe6c3/imagenet/main.py#L234

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')