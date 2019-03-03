# -*- coding: utf-8 -*-
"""
This script should take the disc_preds and cup_preds and 
(1) upsample 
(2) smooth via ellipse fitting
(3) combine as one
(4) compute cdr 
(5) save to pred directory 4

@author: Nabila Abraham
"""
import os 
import cv2 
import numpy as np 
import pandas as pd 
EPS = 1e-7


def vertical_diameter(binary_segmentation):
    '''
    Get the vertical diameter from a binary segmentation.
    The vertical diameter is defined as the "fattest" area of the binary_segmentation parameter.
    '''
    # turn the variable to boolean, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    
    # get the sum of the pixels in the vertical axis
    vertical_axis_diameter = np.sum(binary_segmentation, axis=0)

    # pick the maximum value
    diameter = np.max(vertical_axis_diameter)

    return float(diameter)



def compute_cdr(segmentation):
    '''
    Compute the vertical cup-to-disc ratio from a given labelling map.
    The vertical cup to disc ratio is defined as here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1722393/pdf/v082p01118.pdf
    '''
    # compute the cup diameter
    cup_diameter = vertical_diameter(segmentation==0)
    # compute the disc diameter
    disc_diameter = vertical_diameter(segmentation<255)

    return cup_diameter / (disc_diameter + EPS)


disc_preds = np.load('disc_preds.npy')
cup_preds = np.load('cup_preds.npy')
num_preds = disc_preds.shape[0]

val_data = np.load('val_imgs.npy')
val_names = np.load('val_imgs_list.npy')

orig_img_rows, orig_img_cols = 1634, 1634

k_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
k_cup = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

cdr_preds = np.zeros((num_preds,1))
final_pred = np.ndarray((num_preds, orig_img_rows, orig_img_cols), dtype=np.uint8)

for i in range(num_preds):
    disc_up = cv2.resize(disc_preds[i], dsize=(orig_img_rows, orig_img_cols))
    cup_up = cv2.resize(cup_preds[i], dsize=(orig_img_rows, orig_img_cols))

    disc_sm = cv2.morphologyEx(disc_up, cv2.MORPH_OPEN, k_disc, iterations=3)
    cup_sm = cv2.morphologyEx(cup_up, cv2.MORPH_OPEN, k_cup, iterations=3)
    
    out = ((1-disc_sm) + (1-cup_sm))
    norm = np.max(out).astype(np.uint8)    
    out = np.round((255*out/norm))
    final_pred[i] = out.astype(np.uint8)

    cdr_preds[i] = compute_cdr(final_pred[i])


pred_dir = 'preds4'
curr_dir = os.getcwd()
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
path = os.path.join(curr_dir, pred_dir)

print('-'*30)
print('Writing to directory...')
for i in range(num_preds):
    name = str(val_names[i]).split(".")[0] + ".bmp"
    cv2.imwrite(os.path.join(path,name),final_pred[i])

np.save('cdr_preds4.npy', cdr_preds)    
output = pd.DataFrame(index=val_names, data=cdr_preds, columns=['Glaucoma Risk'])
output.index.name = 'FileName'
output = output.to_csv('preds_from_seg.csv')
