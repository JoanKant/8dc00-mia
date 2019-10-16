# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:23:54 2019

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
train_labels_matrix = np.empty([test_data.shape[0],num_images]) #isn't used only for plotting



for i in np.arange(num_images):
    
    sub = i+1
    train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
    all_data_matrix[:,:,i] = train_data
    train_labels_matrix[:,i] = train_labels.flatten()

#select certain data: 
train_data_matrix= all_data_matrix[:,:,train_subjects]
train_data_matrix = train_data_matrix[:,features,:]
test_data = test_data[:,features]

#Combined K-means
pred_labels_kmeans = np.empty([train_data_matrix.shape[0],train_data_matrix.shape[2]])
print(train_data.shape)
for i in range(train_data_matrix.shape[2]):
    train_data, test_data = seg.normalize_data(train_data_matrix[:,:, i], test_data)

    _, _, w_final = prj.kmeans(train_data, train_labels_matrix[:,i], num_iter, mu)

    temp_pred = prj.predicted_kmeans_test(w_final, test_data)
    
    print("Possible classes are: {}".format(np.unique(temp_pred)))
    tempdice = util.dice_multiclass(test_labels, temp_pred)
    temperr = util.classification_error(test_labels, temp_pred)
    
    print('Err {:.4f}, dice {:.4f}'.format(temperr, tempdice))
    pred_labels_kmeans[:,i] = temp_pred

#decision fusion
predicted_labels_kmeans_final = scipy.stats.mode(pred_labels_kmeans, axis = 1)[0].flatten()

#do a check which labels exist
print("Possible classes are: {}".format(np.unique(predicted_labels_kmeans_final)))

#calculate the error and dice
err = util.classification_error(test_labels, predicted_labels_kmeans_final)
dice = util.dice_multiclass(test_labels, predicted_labels_kmeans_final)

#needed for plotting the real labels and the predicted labels
test_shape = test_labels.reshape(im_size[0],im_size[1])

predicted_mask = predicted_labels_kmeans_final.reshape(im_size[0],im_size[1])
fig, ax = plt.subplots()
ax.imshow(test_shape, 'gray')
 
ax.imshow(predicted_mask, 'viridis', alpha=0.5)
text_str = 'Err {:.4f}, dice {:.4f}'.format(err, dice)
ax.set_xlabel(text_str)
ax.set_title('Subject {}: combined Kmeans'.format(test_subject+1))