# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:06:59 2019

@author: 20171880
"""
import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import suppFunctionSegmentation as sup
import segmentation_project as prj
## Section 1: Single method testing
print("Single method testing: combined kmeans, combined kNN and combined atlases")

#Chosen setttings
train_subjects = [0,1,3,4] 
test_subject = 2
train_slice = 1
test_slice = 1
task = 'tissue'
features = [8,9] #[1,4] and #[0,4] and [8,9]

#DO NOT CHANGE
im_size = [240,240]
sub = test_subject+1
num_images = 5

#the test data and the shape of the labels
test_data, test_labels, test_feature_labels = util.create_dataset(test_subject+1,train_slice,task)
test_shape = test_labels.reshape(im_size[0],im_size[1])

#predefine shapes of the all data and labels
all_data_matrix = np.empty([test_data.shape[0],test_data.shape[1],num_images])
train_labels_matrix = np.empty([test_data.shape[0],num_images]) 

for i in np.arange(num_images):
    sub = i+1
    train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
    all_data_matrix[:,:,i] = train_data
    train_labels_matrix[:,i] = train_labels.flatten()

#select certain data (our selected features): 
train_data_matrix= all_data_matrix[:,:,train_subjects]
train_data_matrix = train_data_matrix[:,features,:]
test_data = test_data[:,features]
train_labels_matrix = train_labels_matrix[:, train_subjects]

#segmentation
mask_cAt, err_cAt, dice_cAt = sup.TestcombinedAtlases(train_labels_matrix, test_labels)
mask_ckmeans, err_kmeans,dice_kmeans =  sup.TestCombinedKmeans(train_data_matrix, train_labels_matrix, test_data, test_labels)
mask_cknn, err_cknn, dice_cknn = sup.TestcombinedkNN(train_data_matrix, train_labels_matrix, test_data, test_labels)


#plotting results
fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(131)
ax1.imshow(test_shape, 'gray')
ax1.imshow(mask_cAt, 'viridis', alpha=0.5)
text_str = 'Err {:.4f}, dice {:.4f}'.format(err_cAt, dice_cAt)
ax1.set_xlabel(text_str)
ax1.set_title('Subject {}: Combined atlas'.format(sub))

ax2 = fig.add_subplot(132)
ax2.imshow(test_shape, 'gray')
ax2.imshow(mask_ckmeans, 'viridis', alpha=0.5)
text_str = 'Err {:.4f}, dice {:.4f}'.format(err_kmeans, dice_kmeans)
ax2.set_xlabel(text_str)
ax2.set_title('Subject {}: Combined k-NN'.format(sub))

ax3 = fig.add_subplot(133)
ax3.imshow(test_shape, 'gray')
ax3.imshow(mask_cknn, 'viridis', alpha=0.5)
text_str = 'Err {:.4f}, dice {:.4f}'.format(err_cknn, dice_cknn)
ax3.set_xlabel(text_str)
ax3.set_title('Subject {}: My method'.format(sub))

##Section 2: Comparing our method with combinedKNN and combined Atlases using the segmentation_demo
#Everything needed is already in the demo, so only running is needed
print("Compare methods, using the segmentation_demo")
prj.segmentation_demo()