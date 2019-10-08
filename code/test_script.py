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


import numpy as np
import scipy
from sklearn.neighbors import KNeighborsClassifier
import random
import segmentation_util as util

import matplotlib.pyplot as plt
import segmentation as seg
from scipy import ndimage, stats


import cv2




train_subject = 1
test_subject = 2
train_slice = 1
test_slice = 1
task = 'brain'

#Load data
train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)

util.scatter_data(train_data, train_labels, 0, 6)
util.scatter_data(test_data, test_labels, 0,6)
