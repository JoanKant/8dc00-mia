# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:42:11 2019

@author: 20171880
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import  segmentation_util as util
import segmentation as seg
from segmentation_tests import funX
import scipy
from sklearn.neighbors import KNeighborsClassifier
from scipy import ndimage, stats

import segmentation_tests as test


#------------------------------------------------------------------#
# TODO: Use the provided code to generate training and testing data
#  Classify the points in test_data, based on their distances d to the points in train_data
train_data, trainlabels = seg.generate_gaussian_data(2)
test_data, testlabels = seg.generate_gaussian_data(1)

D = scipy.spatial.distance.cdist(test_data, train_data, metric='euclidean') #distances between X and C
min_index = np.argmin(D, axis=1)
min_dist = np.zeros((len(min_index),1))
for i in range(len(min_index)):
    min_dist[i,0] = D.item((i, min_index[i]))

# Sort by intensity of cluster center
sorted_order = np.argsort(train_data[:,0], axis=0)

# Update the cluster indices based on the sorted order and return results in
# predicted_labels
predicted_labels = np.empty(*min_index.shape)
predicted_labels[:] = np.nan

for i in np.arange(len(sorted_order)):
    predicted_labels[min_index==sorted_order[i]] = i