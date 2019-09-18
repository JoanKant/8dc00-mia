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



# convert to double and remember the original input type
#
#I = plt.imread('../data/cameraman.tif')
#
## 45 deg. rotation around the image center
#T_1 = util.t2h(reg.identity(), 128*np.ones(2))
#T_2 = util.t2h(reg.rotate(np.pi/4), np.zeros(2))
#T_3 = util.t2h(reg.identity(), -128*np.ones(2))
#T_rot = T_1.dot(T_2).dot(T_3)
#
## 45 deg. rotation around the image center followed by shearing
#T_shear = util.t2h(reg.shear(0.0, 0.5), np.zeros(2)).dot(T_rot)
#
## scaling in the x direction and translation
#T_scale = util.t2h(reg.scale(1.5, 1), np.array([10,20]))
#
#It1, Xt1 = reg.image_transform(I, T_rot)
#It2, Xt2 = reg.image_transform(I, T_shear)
#It3, Xt3 = reg.image_transform(I, T_scale)
#
#fig = plt.figure(figsize=(12,5))
#
#ax1 = fig.add_subplot(131)
#im11 = ax1.imshow(I)
#im12 = ax1.imshow(It1, alpha=0.7)
#
#ax2 = fig.add_subplot(132)
#im21 = ax2.imshow(I)
#im22 = ax2.imshow(It2, alpha=0.7)
#
#ax3 = fig.add_subplot(133)
#im31 = ax3.imshow(I)
#im32 = ax3.imshow(It3, alpha=0.7)
#
#ax1.set_title('Rotation')
#ax2.set_title('Shearing')
#ax3.set_title('Scaling')



X = util.test_object(1)
print(X)
# convert to homogeneous coordinates
Xh = util.c2h(X)

T_rot = reg.rotate(np.pi/4)
T_scale = reg.scale(1.2, 0.9)
T_shear = reg.shear(0.2, 0.1)

T = util.t2h(T_rot.dot(T_scale).dot(T_shear), [10, 20])

Xm = T.dot(Xh)
print(Xm)
Te = reg.ls_affine(Xh, Xm)
print(Te)
Xmt = Te.dot(Xm);

fig = plt.figure(figsize=(12,5))