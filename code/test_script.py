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

X, Y = test.scatter_data_test(showFigs=False)
I = plt.imread('../data/dataset_brains/1_1_t1.tif')
c, coord_im = seg.extract_coordinate_feature(I)
X_data = np.concatenate((X, c), axis=1)
#------------------------------------------------------------------#
# TODO: Write code to normalize your dataset containing variety of features,
#  then examine the mean and std dev
train_data, _ = seg.normalize_data(X_data)
for i in range(6):
   
    norm_feature_mean = np.mean(train_data[i])
    norm_feature_dev = np.std(train_data[i])

    print("The mean is: {:.2f} and the standard deviation is: {:.2f}".format(norm_feature_mean, norm_feature_dev))
    
    #------------------------------------------------------------------#