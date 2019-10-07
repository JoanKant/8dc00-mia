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


#Generates some toy data in 2D, computes PCA, and plots both datasets
N=100
mu1=[0,0]
mu2=[2,0]
sigma1=[[2,1],[1,1]]
sigma2=[[2,1],[1,1]]

XG, YG = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)

util.scatter_data(XG,YG,ax=ax1)
sigma = np.cov(XG, rowvar=False)
w, v = np.linalg.eig(sigma)
ax1.plot([0, v[0,0]], [0, v[1,0]], c='g', linewidth=3, label='Eigenvector1')
ax1.plot([0, v[0,1]], [0, v[1,1]], c='k', linewidth=3, label='Eigenvector2')
ax1.set_title('Original data')
ax_settings(ax1)

ax2 = fig.add_subplot(122)
X_pca, v, w, fraction_variance = seg.mypca(XG)
util.scatter_data(X_pca,YG,ax=ax2)
sigma2 = np.cov(X_pca, rowvar=False)
w2, v2 = np.linalg.eig(sigma2)
ax2.plot([0, v2[0,0]], [0, v2[1,0]], c='g', linewidth=3, label='Eigenvector1')
ax2.plot([0, v2[0,1]], [0, v2[1,1]], c='k', linewidth=3, label='Eigenvector2')
ax2.set_title('My PCA')
ax_settings(ax2)

handles, labels = ax2.get_legend_handles_labels()
plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), bbox_transform=plt.gcf().transFigure, ncol = 4)

print(fraction_variance)
