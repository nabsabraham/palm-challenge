# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:53:15 2018

@author: Nabilla Abraham
"""
import os 
import cv2
import numpy as np 
from skimage import color, filters
import matplotlib.pyplot as plt 



def load_resize_train(img_path, img_list, label,size):
    num_images = len(img_list)
    train = np.ndarray(shape=(num_images,size,size,3), dtype=np.uint8)
    gt = np.ndarray(shape=(num_images, 1))

    for i in range(num_images):
        img = plt.imread(os.path.join(img_path,img_list[i]))
        im = cv2.resize(img, dsize=(size,size), 
                        interpolation=cv2.INTER_CUBIC)
        train[i] = im
        gt[i] = label
    return train, gt 

train_dir = "train\\Training400\\"
val_dir = "validation\\REFUGE-Validation400"
masks_dir = "masks\\Disc_Cup_Masks\\"
currdir = os.getcwd()

pos_path = os.path.join(currdir,train_dir,'Glaucoma')
neg_path = os.path.join(currdir,train_dir,'Non-Glaucoma')

# get pos and neg training images  
pos_imgs = os.listdir(pos_path)
neg_imgs = os.listdir(neg_path)

num_pos = len(pos_imgs)
num_neg = len(neg_imgs)

pos_masks_path = os.path.join(currdir, masks_dir, 'Glaucoma')
neg_masks_path = os.path.join(currdir, masks_dir, 'Non-Glaucoma')

# get pos and neg masks
pos_masks_imgs = os.listdir(pos_masks_path)    
neg_masks_imgs = os.listdir(neg_masks_path)    

train = 

train = np.load('train_imgs.npy')
train_new = color.rgb2lab(train)
plt.imshow(train_new[242,:,:,1], cmap='gray')

img = train_new[100,:,:,1]
idx = np.random.randint(0,400)
value = filters.threshold_local(train_new[idx,:,:,1],101)
im = img>value
plt.imshow(im, cmap='gray')