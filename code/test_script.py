# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:42:11 2019

@author: 20171880
"""


import numpy as np
import matplotlib.pyplot as plt
import  segmentation_util as util

import cv2




train_subject = 1
test_subject = 2
train_slice = 1
test_slice = 1
task = 'brain'

##Load data
#train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
#test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)
##
#util.scatter_data(train_data, train_labels, 0,4)
#util.scatter_data(test_data, test_labels, 0,4)



base_dir = '../data/dataset_brains/'
full_path = base_dir + str(1) + '_' + str(1) + '_t1.tif'
t1_gray = cv2.cvtColor(cv2.imread(full_path),cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(t1_gray, 180, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

t1 = plt.imread(base_dir + str(1) + '_' + str(1) + '_t1.tif')

img = cv2.imread(full_path, 0)
equ = cv2.equalizeHist(img)

cv2.imwrite('equ.png',equ)
readres = plt.imread('equ.png')


kernel1 = np.ones((3,3), np.uint8)
kernel2 =  np.ones((4,4), np.uint8)
kernel3 =   np.ones((3,3), np.uint8)


opening = cv2.morphologyEx(t1, cv2.MORPH_OPEN, kernel1)
closing = cv2.morphologyEx(t1, cv2.MORPH_CLOSE, kernel2)
gradient = cv2.morphologyEx(t1, cv2.MORPH_GRADIENT, kernel3)
cont = cv2.drawContours(t1.copy(), contours, -1, (0,255,0), 3)

#use to remove noise from images, while preserving edges
median = cv2.medianBlur(t1,5)
plt.imshow(median)

fig = plt.figure(figsize = (12,10))
#original
ax1 = fig.add_subplot(231)
ax1.imshow(t1)

#closing
ax2 = fig.add_subplot(232)
ax2.imshow(closing)

#opening
ax3 = fig.add_subplot(233)
ax3.imshow(opening)

#morphological gradient
ax3 = fig.add_subplot(234)
ax3.imshow(gradient)

#contours
ax4 = fig.add_subplot(235)
ax4.imshow(cont)

#equalized histogram image
ax5 = fig.add_subplot(236)
ax5.imshow(readres)

#median

