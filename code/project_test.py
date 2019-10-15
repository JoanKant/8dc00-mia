# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:54:26 2019

@author: 20171880
"""
import numpy as np
import cad_util as util
import matplotlib.pyplot as plt
import registration as reg
import cad
import scipy
from IPython.display import display, clear_output
import scipy.io
import suppFunctionsCAD as sup

fn = './data/nuclei_data.mat'
mat = scipy.io.loadmat(fn)
test_images = mat["test_images"] # shape (24, 24, 3, 20730)
test_y = mat["test_y"] # shape (20730, 1)
training_images = mat["training_images"] # shape (24, 24, 3, 21910)
training_y = mat["training_y"] # shape (21910, 1)

montage_n = 300
sort_ix = np.argsort(training_y, axis=0)
sort_ix_low = sort_ix[:montage_n] # get the 300 smallest
sort_ix_high = sort_ix[-montage_n:] #Get the 300 largest

# visualize the 300 smallest and the 300 largest nuclei
X_small = training_images[:,:,:,sort_ix_low.ravel()]
X_large = training_images[:,:,:,sort_ix_high.ravel()]
fig = plt.figure(figsize=(16,8))
ax1  = fig.add_subplot(121)
ax2  = fig.add_subplot(122)
util.montageRGB(X_small, ax1)
ax1.set_title('300 smallest nuclei')
util.montageRGB(X_large, ax2)
ax2.set_title('300 largest nuclei')

# dataset preparation
imageSize = training_images.shape

# every pixel is a feature so the number of features is:
# height x width x color channels
numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
training_x = training_images.reshape(numFeatures, imageSize[3]).T.astype(float)
test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

## training linear regression model
#---------------------------------------------------------------------#
# TODO: Implement training of a linear regression model for measuring
# the area of nuclei in microscopy images. Then, use the trained model
# to predict the areas of the nuclei in the test dataset.
_, _, predicted_y = sup.linear_regression(training_x, test_x)
#---------------------------------------------------------------------#

# visualize the results
fig2 = plt.figure(figsize=(16,8))
ax1  = fig2.add_subplot(121)
line1, = ax1.plot(predicted_y, test_y, ".g", markersize=3)
ax1.grid()
ax1.set_xlabel('Area')
ax1.set_ylabel('Predicted Area')
ax1.set_title('Training with full sample')

#training with smaller number of training samples
#---------------------------------------------------------------------#
# TODO: Train a model with reduced dataset size (e.g. every fourth
# training sample).
#---------------------------------------------------------------------#

# visualize the results
ax2  = fig2.add_subplot(122)
line2, = ax2.plot(predicted_y, test_y, ".g", markersize=3)
ax2.grid()
ax2.set_xlabel('Area')
ax2.set_ylabel('Predicted Area')
ax2.set_title('Training with smaller sample')