# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:55:04 2019

@author: 20171880
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output
import registration_adapted_functions as reg_adapt


# initial values for the parameters
# we start with the identity transformation
# most likely you will not have to change these

# NOTE: for affine registration you have to initialize
# more parameters and the scaling parameters should be
# initialized to 1 instead of 0

# the similarity function
# this line of code in essence creates a version of rigid_corr()
# in which the first two input parameters (fixed and moving image)
# are fixed and the only remaining parameter is the vector x with the
# parameters of the transformation
I1_path = '../data/image_data/1_1_t1.tif'
Im1_path = '../data/image_data/1_1_t1_d.tif'
I2_path = '../data/image_data/1_1_t2.tif'

I = plt.imread(I1_path)
Im = plt.imread(Im1_path)

fun = lambda x: reg_adapt.rigid_corr_adapted(I, Im, x)
x = np.array([0.,0.,0.])
h = 1e-3
length_x = len(x)
g = (np.zeros((1,length_x)))     
for i in range( length_x):
    inputparameters_1 = x.copy()
    inputparameters_2 = x.copy()
    print(inputparameters_1)
    inputparameters_1[i] = x[i]+h/2
    print(inputparameters_1)
    inputparameters_2[i] = x[i]-h/2
    print(inputparameters_2)
    counter = np.subtract(fun(inputparameters_1),fun(inputparameters_2))
    
    print(counter)
    g[0,i] = (counter/h)