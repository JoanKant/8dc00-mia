# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:53:51 2019

@author: 20171880
"""
import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output
import registration_util as util
import registration_adapted_functions as reg_adapt
import registration_project as proj

I1_path = '../data/image_data/1_1_t1.tif'
Im1_path = '../data/image_data/1_1_t1_d.tif'
I2_path = '../data/image_data/1_1_t2.tif'

I_path = I1_path
Im_path = Im1_path

imageI = plt.imread(I_path);
imageIm = plt.imread(Im_path);

#convert to homogenous coordinates using c2h
numberofpoints = 8 #change to 4, 8 or 12
set_of_images = 2# 1= T1 and T1m and 2 = T1 and T2
path = "C:/Users/20171880/Desktop/8dc00-mia/results/resultsTransformationMatrix" #change to your path to resultsTransformationMatrix


if set_of_images == 1:
    full_path = path+"/transformation_matrix_T1_and_T1_moving_"+str(numberofpoints)+".npy"
    

else:         
    full_path = path+"/transformation_matrix_T1_and_T2_"+str(numberofpoints)+".npy"
    

#compute affine transformation and make a homogenous transformation matrix
Th  = np.load(full_path) 





#transfrom the moving image using the transformation matrix
It, Xt = reg.image_transform(imageIm, Th)

#plotting the results
fig = plt.figure(figsize = (20,30))

ax1 = fig.add_subplot(131)
im1 = ax1.imshow(imageI) #plot first image (fixed or T1) image
ax2 = fig.add_subplot(132)
im2 = ax2.imshow(imageIm) #plot second image (moving or T2) image
ax3 = fig.add_subplot(133)
im3 = ax3.imshow(It) #plot (T1 moving or T2) transformed image

if set_of_images == 1:
    ax1.set_title('T1')
    ax2.set_title('T1 moving')
    ax3.set_title('Transformed T1 moving')

else:
    ax1.set_title('T1')
    ax2.set_title('T2')
    ax3.set_title('Transformed T2')