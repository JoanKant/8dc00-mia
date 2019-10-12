# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:15:59 2019

@author: 20171880
"""


import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg
from IPython.display import display, clear_output
from scipy import ndimage, stats
import scipy
import segmentation_project as prj

train_subjects = [0,1,3,4] 
test_subject = 2
train_slice = 1
test_slice = 1
task = 'tissue'
im_size = [240, 240]
num_images = 5
features = [8,9] #[1,4] and #[0,4] and [8,9]
num_iter = 100
mu = 0.1
test_data, test_labels, test_feature_labels = util.create_dataset(test_subject+1,train_slice,task)

all_data_matrix = np.empty([test_data.shape[0],test_data.shape[1],num_images])
train_labels_matrix = np.empty([test_data.shape[0],num_images])

for i in np.arange(num_images):
    sub = i+1
    train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
    all_data_matrix[:,:,i] = train_data
    train_labels_matrix[:,i] = train_labels.flatten()

#select certain data: 
train_data_matrix= all_data_matrix[:,:,train_subjects]
train_data_matrix = train_data_matrix[:,features,:]
test_data = test_data[:,features]
train_labels_matrix = train_labels_matrix[:, train_subjects]

#predict test data labels
predicted_labels, predicted_labels2_atlas = seg.segmentation_combined_atlas(train_labels_matrix)
predicted_labels,predicted_labels2_knn = seg.segmentation_combined_knn(train_data_matrix,train_labels_matrix,test_data)

#calculate error and dice 
dice_atlas = util.dice_multiclass(test_labels, predicted_labels2_atlas)
err_atlas = util.classification_error(test_labels, predicted_labels2_atlas)

dice_knn = util.dice_multiclass(test_labels, predicted_labels2_knn)
err_knn = util.classification_error(test_labels, predicted_labels2_knn)

#needed for plotting the 'real' data and the predicted 
test_shape = test_labels.reshape(im_size[0],im_size[1])

#Plot for combined atlas
predicted_mask_atlas = predicted_labels2_atlas.reshape(im_size[0],im_size[1])
fig, ax = plt.subplots()
ax.imshow(test_shape, 'gray')
 
ax.imshow(predicted_mask_atlas, 'viridis', alpha=0.5)
text_str = 'Err {:.4f}, dice {:.4f}'.format(err_atlas, dice_atlas)
ax.set_xlabel(text_str)
ax.set_title('Subject {}: combined Atlases'.format(test_subject+1))


#Plo1t for combined knn
predicted_mask_atlas = predicted_labels2_knn.reshape(im_size[0],im_size[1])
fig2, ax2 = plt.subplots()
ax2.imshow(test_shape, 'gray')
# 
ax2.imshow(predicted_mask_atlas, 'viridis', alpha=0.5)
text_str = 'Err {:.4f}, dice {:.4f}'.format(err_knn, dice_knn)
ax2.set_xlabel(text_str)
ax2.set_title('Subject {}: combined KNN'.format(test_subject+1))