# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:02:42 2019

@author: 20171880
"""
import numpy as np
import segmentation as seg 
import segmentation_util as util
import segmentation_project as prj
import scipy

def TestCombinedKmeans(train_data_matrix, train_labels_matrix, test_data, test_labels):
    im_size = [240, 240]

    #Combined K-means
    pred_labels_kmeans = np.empty([train_data_matrix.shape[0],train_data_matrix.shape[2]])
    for i in range(train_data_matrix.shape[2]):
        #normalize data
        train_data, test_data = seg.normalize_data(train_data_matrix[:,:, i], test_data)
        #get optimized clusters (using 100 iterations and a learning rate of 0.1)
        _, _, w_final = prj.kmeans_no_plot(train_data, train_labels_matrix[:,i], 100, 0.1)
        #predict the labels
        temp_pred = prj.predicted_kmeans_test(w_final, test_data)
        
        #store labels for each training subject in one matrix
        pred_labels_kmeans[:,i] = temp_pred
    
    #decision fusion based on majority voting
    predicted_labels_kmeans_final = scipy.stats.mode(pred_labels_kmeans, axis = 1)[0].flatten()
    
    #calculate the error and dice
    err = util.classification_error(test_labels, predicted_labels_kmeans_final)
    dice = util.dice_multiclass(test_labels, predicted_labels_kmeans_final)
    predicted_mask = predicted_labels_kmeans_final.reshape(im_size[0],im_size[1])

    return predicted_mask, err, dice

def TestcombinedkNN(train_data_matrix, train_labels_matrix, test_data, test_labels):
    im_size = [240, 240]

    #predict test data labels
    predicted_labels,predicted_labels2_knn = seg.segmentation_combined_knn(train_data_matrix,train_labels_matrix,test_data)
    #calculate error and dice 
    dice_knn = util.dice_multiclass(test_labels, predicted_labels2_knn)
    err_knn = util.classification_error(test_labels, predicted_labels2_knn)
    predicted_mask_atlas = predicted_labels2_knn.reshape(im_size[0],im_size[1])
    return  predicted_mask_atlas,err_knn, dice_knn

def TestcombinedAtlases(train_labels_matrix, test_labels):
    im_size = [240, 240]
    #predict the test data labels
    predicted_labels, predicted_labels2_atlas = seg.segmentation_combined_atlas(train_labels_matrix)
    #calculate error and dice 
    dice_atlas = util.dice_multiclass(test_labels, predicted_labels2_atlas)
    err_atlas = util.classification_error(test_labels, predicted_labels2_atlas)
    predicted_mask_atlas = predicted_labels2_atlas.reshape(im_size[0],im_size[1])
    return predicted_mask_atlas, err_atlas, dice_atlas, 
