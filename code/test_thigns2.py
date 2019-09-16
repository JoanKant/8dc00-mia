# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:00:14 2019

@author: 20171880
"""
import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output


I = plt.imread('../data/cameraman.tif')

# 45 deg. rotation around the image center
T_1 = util.t2h(reg.identity(), 128*np.ones(2))
T_2 = util.t2h(reg.rotate(np.pi/4), np.zeros(2))
T_3 = util.t2h(reg.identity(), -128*np.ones(2))
T_rot = T_1.dot(T_2).dot(T_3)

# 45 deg. rotation around the image center followed by shearing
T_shear = util.t2h(reg.shear(0.0, 0.5), np.zeros(2)).dot(T_rot)

# scaling in the x direction and translation
T_scale = util.t2h(reg.scale(1.5, 1), np.array([10,20]))
#
input_type = type(I);

# default output size is same as input
if output_shape is None:
    output_shape = I.shape

# spatial coordinates of the transformed image
#x = np.arange(0, output_shape[1])
#y = np.arange(0, output_shape[0])
#xx, yy = np.meshgrid(x, y)
#
## convert to a 2-by-p matrix (p is the number of pixels)
#X = np.concatenate((xx.reshape((1, xx.size)), yy.reshape((1, yy.size))))
## convert to homogeneous coordinates
#Xh = util.c2h(X)
#
##------------------------------------------------------------------#
## TODO: Perform inverse coordinates mapping.
#Xt = np.linalg.pinv(Th)
#
##------------------------------------------------------------------#
#
#It = ndimage.map_coordinates(I, [Xt[1,:], Xt[0,:]], order=1, mode='constant').reshape(I.shape)
