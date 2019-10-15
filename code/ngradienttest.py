# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:24:02 2019

@author: 20171880
"""
from builtins import print
import numpy as np
import cad_util as util
import segmentation_util as seg_util
import registration as reg
import segmentation as seg
import matplotlib.pyplot as plt
import cad
from IPython.display import display, clear_output, HTML
import numpy as np
num_training_samples = 300
num_validation_samples = 100

# here we reuse the function from the segmentation practicals
m1=[2,3]
m2=[-0,-4]
s1=[[8,7],[7,8]]
s2=[[8,6],[6,8]]

[trainingX, trainingY] = seg.generate_gaussian_data(num_training_samples, m1, m2, s1, s2)
r,c = trainingX.shape
print('Training sample shape: {}'.format(trainingX.shape))

# we need a validation set to monitor for overfitting
[validationX, validationY] = seg.generate_gaussian_data(num_validation_samples, m1, m2, s1, s2)
r_val,c_val = validationX.shape
print('Validation sample shape: {}'.format(validationX.shape))

validationXones = util.addones(validationX)

# train a logistic regression model:
# the learning rate for the gradient descent method
# (the same as in intensity-based registration)
mu = 0.001

# we are actually using stochastic gradient descent
batch_size = 30

# initialize the parameters of the model with small random values,
# we need one parameter for each feature and a bias
Theta = 0.02*np.random.rand(c+1, 1)

# number of gradient descent iterations
num_iterations = 300

# variables to keep the loss and gradient at every iteration
# (needed for visualization)
iters = np.arange(num_iterations)
loss = np.full(iters.shape, np.nan)
validation_loss = np.full(iters.shape, np.nan)

# pick a batch at random
idx = np.random.randint(r, size=batch_size)

# the loss function for this particular batch
fun = lambda Theta: cad.lr_nll(util.addones(trainingX[idx,:]), trainingY[idx], Theta)
 

x = Theta
h =1e-3
# Computes the derivative of a function with numerical differentiation.
# Input:
# fun - function for which the gradient is computed
# x - vector of parameter values at which to compute the gradient
# h - a small positive number used in the finite difference formula
# Output:
# g - vector of partial derivatives (gradient) of fun

#------------------------------------------------------------------#
# TODO: Implement the  computation of the partial derivatives of
# the function at x with numerical differentiation.
# g[k] should store the partial derivative w.r.t. the k-th parameter
length_x = len(x)
if (length_x == 1):
    counter = fun(x[0]+h/2)-fun(x[0]-h/2)
    g = counter/h
else:  #several partial derivatives
    g = (np.zeros((1,length_x)))     
    for i in range(length_x):
        inputparameters_1 = x.copy()
        inputparameters_2 = x.copy()
        inputparameters_1[i] = x[i]+h/2
        inputparameters_2[i] = x[i]-h/2
        counter = np.subtract(fun(inputparameters_1),fun(inputparameters_2))
        g[0,i] = (counter/h)
#------------------------------------------------------------------#
