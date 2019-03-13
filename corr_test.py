#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:45:07 2019

@author: nabila
"""
import torch
from torchvision import models, transforms
from sklearn.cross_decomposition import CCA

X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
cca = CCA(n_components=1)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)

vgg = models.vgg16(pretrained=True)

features_detector = vgg.features
