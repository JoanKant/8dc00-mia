# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:06:59 2019

@author: 20171880
"""
import segmentation_util as util
import segmentation as seg
import segmentation_project as prj
import numpy as np

# Load data
train_subject = 1
test_subject = 2
train_slice = 1
test_slice = 1
task = 'tissue'

#load data for method: kmeans and combined nearest neighbour
train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)
 
#DO NOT CHANGE the following section
#load data for atlas (atlases do not use testdata [look at description of the function in segmentation])
num_images = 5
num_methods = 3
im_size = [240, 240]

all_subjects = np.arange(num_images)
task = 'tissue'
all_data_matrix = np.empty([train_data.shape[0],train_data.shape[1],num_images])
all_labels_matrix = np.empty([train_labels.size,num_images], dtype=bool)
all_subjects = np.arange(num_images)

for i in all_subjects:
    sub = i+1
    train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
    all_data_matrix[:,:,i] = train_data
    all_labels_matrix[:,i] = train_labels.flatten()
    
#Select data with certain features and normalize it
features = [4,5] #you can change it to whatever features you want
train_data,_ = seg.normalize_data(train_data[:, features])
test_data ,_= seg.normalize_data(test_data[:,features])
all_data_matrix, _ = seg.normalize_data(all_data_matrix[:, features, :])

#Use my method (in the method the three 'submethods' will be combined)
predicted_labels = prj.segmentation_mymethod(train_data, train_labels, all_data_matrix, all_labels_matrix, test_data, task='tissue')

dice = util.dice_multiclass(train_labels, predicted_labels)
error = util.classification_error(train_labels, predicted_labels)

print("Dice score of combined methods is {:.2f}".format(dice))
print("Error of combined methods is {:.2f}".format(error))


