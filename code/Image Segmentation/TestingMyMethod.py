# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:45:56 2019

@author: 20171880
"""
import segmentation_util as util
import segmentation as seg
import segmentation_project as prj
import numpy as np
import scipy

# Load data
train_subject = 1
test_subject = 2
train_slice = 1
test_slice = 1
task = 'tissue'


train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)
 
num_images = 5
num_methods = 3
im_size = [240, 240]

#all_errors = np.empty([num_images,num_methods])
#all_errors[:] = np.nan
#all_dice = np.empty([num_images,num_methods])
#all_dice[:] = np.nan

all_subjects = np.arange(num_images)
train_slice = 1
task = 'tissue'
all_data_matrix = np.empty([train_data.shape[0],train_data.shape[1],num_images])
all_labels_matrix = np.empty([train_labels.size,num_images], dtype=bool)


for i in all_subjects:
    sub = i+1
    train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
    all_data_matrix[:,:,i] = train_data
    all_labels_matrix[:,i] = train_labels.flatten()

#Select data with certain features and normalize it
features = [1,4]
train_data,_ = seg.normalize_data(train_data[:, features])
test_data ,_= seg.normalize_data(test_data[:,features])
all_data_matrix, _ = seg.normalize_data(all_data_matrix[:, features, :])


#predicted_train = seg.kmeans_clustering(train_data, K=4)
if (task == 'tissue'):
    k = 4
else:
    k = 2
    
kmeans_cost, train_predicted, w_final =  prj.kmeans(train_data, train_labels k, mu = 0.1, num_iter = 5)

dice = util.dice_multiclass(train_labels, train_predicted)
error = util.classification_error(train_labels, train_predicted)

print("Dice score is {:.2f}".format(dice))
print("Error is {:.2f}".format(error))

#Use my method
#predicted_labels = prj.segmentation_mymethod(train_data, train_labels, test_data, task='tissue')
pred_labels_kmeans = prj.predicted_kmeans_test(w_final, test_data).T

_, pred_labels_cat = seg.segmentation_combined_atlas(train_labels, combining='mode')
_, pred_labels_cnn = seg.segmentation_combined_knn(all_data_matrix, all_labels_matrix, test_data, k=1)

pred_labels_cat = pred_labels_cat.T
pred_labels_cnn = pred_labels_cnn.T

concat_labels = np.vstack((pred_labels_kmeans, pred_labels_cat, pred_labels_cnn)).T
predicted_labels = scipy.stats.mode(concat_labels, axis = 1)[0]
    

dice = util.dice_multiclass(train_labels, predicted_labels)
error = util.classification_error(train_labels, predicted_labels)

print("Dice score of combined methods is {:.2f}".format(dice))
print("Error of combined methods is {:.2f}".format(error))

