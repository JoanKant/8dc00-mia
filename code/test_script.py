# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:42:11 2019

@author: 20171880
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
I = plt.imread('../data/dataset_brains/1_1_t1.tif')
import  segmentation_util as util
import segmentation as seg

import segmentation_tests as test




# TODO: plot the datasets on top of each other in different colors and visualize the data,
#  calculate the distances between the datasets,
#  order the distances (min to max) using the provided code,
#  calculate how many samples are closest to each of the samples in `C`



#plotting X and C on top of each other

plt.scatter(X[:,0], X[:,1], color = 'b')
plt.scatter(C[:,0], C[:,1], color = 'r') #X1 and X2

min_index = np.argmin(D, axis=1) #Returns the indices of the minimum values along an axis (vertical axis)
min_dist = D[:,min_index]

print(min_index)
print(min_dist)   

