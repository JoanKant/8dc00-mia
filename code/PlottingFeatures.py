# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:59:32 2019

@author: 20171880
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage, stats
import cv2
import segmentation as seg

image_number = 1; 
slice_number = 1
   # extracts features for [image_number]_[slice_number]_t1.tif and [image_number]_[slice_number]_t2.tif
# Input:
# image_number - Which subject (scalar)
# slice_number - Which slice (scalar)
# Output:
# X           - N x k dataset, where N is the number of pixels and k is the total number of features
# features    - k x 1 cell array describing each of the k features

base_dir = '../data/dataset_brains/'

t1 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t1.tif')

n = t1.shape[0]
features = ()

t1f = t1.flatten().T.astype(float)
t1f = t1f.reshape(-1, 1)

t1_blurred_1 = ndimage.gaussian_filter(t1, sigma=3)
t1_1 = t1_blurred_1.flatten().T
t1_1 = t1_1.reshape(-1, 1)

t1_blurred_2 = ndimage.gaussian_filter(t1, sigma=8)
t1_2 = t1_blurred_2.flatten().T
t1_2 = t1_2.reshape(-1, 1)

#edges
t1_LaPlacian = cv2.Laplacian(t1, cv2.CV_64F)
t1_lapl = t1_LaPlacian.flatten().T
t1_lapl = t1_lapl.reshape(-1,1)

#Coordinate feature
t1_coord,_ = seg.extract_coordinate_feature(t1)


#use to remove noise from images, while preserving edges
median = cv2.medianBlur(t1,5)
t1_median = median.flatten().flatten().T
t1_median = t1_median.reshape(-1,1)

#opening
kernel1 = np.array([[0,1,0], [1, 1, 1], [0,1,0]], np.uint8)#np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(t1, cv2.MORPH_OPEN, kernel1)
t1_opening = opening.flatten().flatten().T
t1_opening = t1_opening.reshape(-1,1)

#closing
kernel2 =  np.ones((4,4), np.uint8)
closing = cv2.morphologyEx(t1, cv2.MORPH_CLOSE, kernel2)
t1_closing = closing.flatten().flatten().T
t1_closing = closing.reshape(-1,1)

#morphological gradient
kernel3 =   np.ones((3,3), np.uint8)
gradient = cv2.morphologyEx(t1, cv2.MORPH_GRADIENT, kernel3)
t1_grad = gradient.flatten().flatten().T
t1_grad = t1_grad.reshape(-1,1)

t1_cv = cv2.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t1.tif', 0)
equ = cv2.equalizeHist(t1_cv)
cv2.imwrite('equ.png',equ)
t1_equ = plt.imread('equ.png')
t1_equ = t1_equ.flatten().T
t1_equ = t1_equ.reshape(-1,1)


#Add features to the list of features
features += ('T1 intensity',)
features += ('T1 gauss 5',)
features += ("T1 equalized histogram", )
features += ('T1 gauss 15',)
features += ("T1 Coordinate feature",)
features += ("T1 median",)
features += ("T1 opening",)
features += ("T1 closing",)
features += ("T1 morphological gradient",)
features += ("T1 Laplacian",)

X = np.concatenate((t1f,t1_equ,  t1_1, t1_1,t1_coord, t1_median, t1_opening, t1_closing, t1_grad, t1_lapl), axis=1)

fig = plt.figure(figsize = (15,20))
ax1 = fig.add_subplot(131)
ax1.imshow(t1)
ax1.set_title('Original image')

ax2 = fig.add_subplot(132)
ax2.imshow(closing)
ax2.set_title('closing')
