# -*- coding: utf-8 -*-
"""
Test script for project Segmentation
"""

import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg
import segmentation_tests as tst
train_subject = 1
test_subject = 2
train_slice = 1
test_slice = 1
task = 'brain'

#Load data
train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)


#normalize data (needed for PCA see the slides)
train_data,_ = seg.normalize_data(train_data[:,4:8])#selecting parts of the features gives also other results

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)

util.scatter_data(train_data,train_labels,ax=ax1)
sigma = np.cov(train_data, rowvar=False)
w, v = np.linalg.eig(sigma)
ax1.plot([0, v[0,0]], [0, v[1,0]], c='g', linewidth=3, label='Eigenvector1')
ax1.plot([0, v[0,1]], [0, v[1,1]], c='k', linewidth=3, label='Eigenvector2')
ax1.set_title('Original data')
tst.ax_settings(ax1)

ax2 = fig.add_subplot(122)
X_pca, v, w, fraction_variance = seg.mypca(train_data)
util.scatter_data(X_pca,train_labels,ax=ax2)
sigma2 = np.cov(X_pca, rowvar=False)
w2, v2 = np.linalg.eig(sigma2)
ax2.plot([0, v2[0,0]], [0, v2[1,0]], c='g', linewidth=3, label='Eigenvector1')
ax2.plot([0, v2[0,1]], [0, v2[1,1]], c='k', linewidth=3, label='Eigenvector2')
ax2.set_title('My PCA')
tst.ax_settings(ax2)
#
handles, labels = ax2.get_legend_handles_labels()
plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), bbox_transform=plt.gcf().transFigure, ncol = 4)



#Order of the features (useful for selecting features)
#features += ('T1 intensity',)
#features += ('T1 gauss 5',)
#features += ("T1 equalized histogram", )
#features += ('T1 gauss 15',)
#features += ("T1 Coordinate feature",)
#features += ("T1 median",)
#features += ("T1 opening",)
#features += ("T1 closing",)
#features += ("T1 morphological gradient",)
#features += ("T1 Laplacian",)
#
#
#
