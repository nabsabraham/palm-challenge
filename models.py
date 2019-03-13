#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 02:34:20 2019

@author: nabila
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel=3, mode="double_conv"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.mode = mode

        if self.mode == "double_conv":
            self.main = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, self.kernel, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(self.out_channels),
                    nn.Conv2d(self.out_channels, self.out_channels, self.kernel, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(self.out_channels)
                    )
        elif self.mode == "single_conv":
            self.main = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, self.kernel, padding=0),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(self.out_channels)
                    )
            
    def forward(self, x):
        return self.main(x)

        

class UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of 
    Upsample -> Concat -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, mode="conv_transpose"):
        super().__init__()


        if mode == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif mode == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
            
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, smaller_map, skip_map):
        """
        :param smaller_map: this is the output from the previous block
        :param skip_map: this is the output skipped from a previous layer
        :return: upsampled feature map 
        """
        x = self.upsample(smaller_map)
        x = torch.cat([x, skip_map], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        
        return x

class UpBlock_noskip(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of 
    Upsample -> Concat -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, mode="conv_transpose"):
        super().__init__()

        if mode == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif mode == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
            
        self.conv_block_1 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        """
        :param x: this is the output from the previous block so the smaller map
        :param skip_map: this is the output skipped from a previous layer
        :return: upsampled feature map 
        """
        x = self.upsample(x)
        x = self.conv_block_1(x)
        
        return x



class unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.conv1 = ConvBlock(self.n_channels, 32)
        self.conv2 = ConvBlock(32, 64, mode="double_conv")
        self.conv3 = ConvBlock(64,128, mode="double_conv")        
        self.conv4 = ConvBlock(128,256, mode="double_conv")
        self.conv5 = ConvBlock(256, 512, mode="double_conv")

        self.maxpool = nn.MaxPool2d(2)
        
        self.up4 = UpBlock(512, 256, mode="conv_transpose") 
        self.up3 = UpBlock(256, 128, mode="conv_transpose") 
        self.up2 = UpBlock(128, 64, mode="conv_transpose")         
        self.up1 = UpBlock(64, 32, mode="conv_transpose") 
        
        self.output = ConvBlock(32, self.n_classes, kernel=1, mode="single_conv")
       

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
        x = self.up4(x, conv4)
        x = self.up3(x, conv3)
        x = self.up2(x, conv2)
        x = self.up1(x, conv1)
        
        out = self.output(x)
        
        return torch.sigmoid(out)

class decoder(nn.Module):
    def __init__(self, input_channels, n_classes):
        super().__init__()
        
        self.up4 = UpBlock(512, 256, mode="conv_transpose") 
        self.up3 = UpBlock(256, 128, mode="conv_transpose") 
        self.up2 = UpBlock(128, 64, mode="conv_transpose")         
        self.up1 = UpBlock(64, 32, mode="conv_transpose") 
        self.output = ConvBlock(32, self.n_classes, kernel=1, mode="single_conv")
        
    def forward(self,x):   
        
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        out = self.output(x)
        return F.sigmoid(out)
    
class vgg_decoder(nn.Module):
    def __init__(self, h_dim, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.fc1 = nn.Linear(h_dim, 4096)
        self.fc2 = nn.Linear(4096, (7*7*256))
        self.act = nn.ReLU()

        self.up5 = UpBlock_noskip(256, 256, mode="conv_transpose")
        self.up4 = UpBlock_noskip(256, 128, mode="conv_transpose")
        self.up3 = UpBlock_noskip(128, 64, mode="conv_transpose")       
        self.up2 = UpBlock_noskip(64, 32, mode="conv_transpose")
        self.up1 = UpBlock_noskip(32, 16, mode="conv_transpose") 
        self.output = ConvBlock(16, self.n_classes, kernel=1, mode="single_conv")
        
    def forward(self,x): 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act(x)
        
        x = x.view(x.shape[0], 256, 7, 7)
        x = self.up5(x)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)

        out = self.output(x)
        return torch.sigmoid(out)
        
if __name__=="__main__":
#    model = unet(3,1)
#    model = model.to('cuda')
#    summary(model, input_size=(3,256,256))
    model = vgg_decoder(1).to('cuda')
    summary(model, input_size=(1,301))
    