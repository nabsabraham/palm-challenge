#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:34:17 2019

@author: nabila
"""
import torch
import torch.nn as nn

def conv_relu_bn(in_channels, out_channels, kernel=3):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
            )
    
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.conv1 = conv_relu_bn(self.n_channels,64)
        self.conv2 = conv_relu_bn(64,128)
        self.conv3 = conv_relu_bn(128,256)
        self.conv4 = conv_relu_bn(256,512)
        self.conv5 = conv_relu_bn(512,512)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.up4 = conv_relu_bn(512 + 512, 512)
        self.up3 = conv_relu_bn(256 + 512, 256)
        self.up2 = conv_relu_bn(128 + 256, 128)
        self.up1 = conv_relu_bn(64 + 128, 64)
        
        self.output = nn.Sequential(nn.Conv2d(64, n_classes, 1),
                                    nn.Sigmoid())
        
    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.conv2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.conv3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.conv4(x)
        x = self.maxpool(conv4)
       
        x = self.conv5(x)
        
        x = self.upsample(x)
        x = torch.cat([conv4, x], dim=1)
        x = self.up4(x)
        
        x = self.upsample(x)
        x = torch.cat([conv3, x], dim=1)
        x = self.up3(x)
        
        x = self.upsample(x)
        x = torch.cat([conv2, x], dim=1)
        x = self.up2(x)
        
        x = self.upsample(x)
        x = torch.cat([conv1, x], dim=1)
        x = self.up1(x)
        out = self.output(x)
        return out



class small_unet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        
        self.conv1 = conv_relu_bn(self.n_channels,64)
        self.conv2 = conv_relu_bn(64,128)
        self.conv3 = conv_relu_bn(128,256)
        self.conv4 = conv_relu_bn(256,512)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.up3 = conv_relu_bn(256 + 512, 256)
        self.up2 = conv_relu_bn(128 + 256, 128)
        self.up1 = conv_relu_bn(64 + 128, 64)
        
        self.output = nn.Sequential(nn.Conv2d(64, 1, 1),
                                    nn.Sigmoid())
        
    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.conv2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.conv3(x)
        x = self.maxpool(conv3)
        
        x = self.conv4(x)
        
        x = self.upsample(x)
        x = torch.cat([conv3, x], dim=1)
        x = self.up3(x)
        
        x = self.upsample(x)
        x = torch.cat([conv2, x], dim=1)
        x = self.up2(x)
        
        x = self.upsample(x)
        x = torch.cat([conv1, x], dim=1)
        x = self.up1(x)
        out = self.output(x)
        return out
   
#from torchsummary import summary
#summary(UNet(3,1).to('cuda'), input_size=(3,144,144))