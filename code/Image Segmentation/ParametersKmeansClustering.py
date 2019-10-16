# -*- coding: utf-8 -*-
"""
Finding values for the learning rate (mu) and iterations to use in kmeans
"""

import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg
from IPython.display import display, clear_output
from scipy import ndimage, stats
import scipy
import segmentation_project as prj
# Load data
train_subject = 1
test_subject = 2
train_slice = 1
test_slice = 1
task = 'tissue'


features = [8,9]
num_iter = 100
mu = 0.1


train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)

num_images = 5
train_slice = 1
task = 'tissue'

#select certain data: 
train_data_matrix = train_data[:,features]



_, _, w_final = prj.kmeans(train_data_matrix, train_labels, num_iter, mu)
    