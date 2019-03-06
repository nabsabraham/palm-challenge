# -*- coding: utf-8 -*-
"""
Metrics and loss functions 

@author: Nabilla Abraham
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def dice_score(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dsc = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    return dsc.mean()

def dice_loss(pred, target, smooth=1):
    dsc = dice_score(pred, target)
    return 1 - dsc
   
def tversky(y_true, y_pred, smooth=1., alpha=0.5):
    y_true_pos = y_true.contiguous()
    y_pred_pos = y_pred.contiguous()
    true_pos = (y_true_pos * y_pred_pos).sum().sum()
    false_neg = (y_true_pos * (1-y_pred_pos)).sum().sum()
    false_pos = (1-y_true_pos)*(y_pred_pos).sum().sum()
    
    return ((true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)).mean()

def tl(y_true, y_pred):
    return 1-tversky(y_true,y_pred, alpha=0.6)

def ftl(y_true, y_pred, gamma=0.75):
    x = 1-tversky(y_true, y_pred, alpha=0.6)
    return x**gamma

def vae_loss(pred, gt, mu, logvar):
    dl = dice_loss(pred,gt)
    kl = -0.001 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl + dl

