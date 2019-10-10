# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 07:32:31 2019

@author: 20171880
"""
import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg
from scipy import ndimage, stats
import scipy
from sklearn.neighbors import KNeighborsClassifier
import timeit
from IPython.display import display, clear_output




N=100
mu1=[0,0]
mu2=[2,0]
sigma1=[[2,1],[1,1]]
sigma2=[[2,1],[1,1]]

XG, YG = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

X_pca, v, w, fraction_variance = seg.mypca(XG)

X = XG
X = X - np.mean(X, axis=0)


#------------------------------------------------------------------#
#TODO: Calculate covariance matrix of X, find eigenvalues and eigenvectors,
# sort them, and rotate X using the eigenvectors
cov_matrix = np.cov(X,rowvar=False)
np.sort(cov_matrix)
w,v   = np.linalg.eig(cov_matrix)
print(w)
print(v)

idx = np.argsort(w)[::-1]
w = w[idx]
v = v[:,idx]
print(w)
print(v)

X_pca = X.dot(v)

#------------------------------------------------------------------#

#Return fraction of variance
fraction_variance = np.zeros((X_pca.shape[1],1))
for i in np.arange(X_pca.shape[1]):
    fraction_variance[i] = np.sum(w[:i+1])/np.sum(w)