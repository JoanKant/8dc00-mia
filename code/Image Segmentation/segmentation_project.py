#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project code+scripts for 8DC00 course
"""

# Imports

import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg
from IPython.display import display, clear_output
from scipy import ndimage, stats
import scipy

def segmentation_mymethod(train_data, train_labels , test_data,num_iter = 100,mu = 0.1):
#    def segmentation_mymethod(train_data, train_labels, all_data_matrix, all_labels_matrix, test_data,num_iter = 200,mu = 0.01, task='tissue'):

    # segments the image based on your own method!
    # Input:
    # train_data_matrix   num_pixels x num_features x num_subjects matrix of
    # features
    # train_labels_matrix num_pixels x num_subjects matrix of labels
    # test_data           num_pixels x num_features test data
    # task           String corresponding to the segmentation task: either 'brain' or 'tissue'
    # Output:
    # predicted_labels    Predicted labels for the test slice

    #------------------------------------------------------------------#
    #TODO: Implement your method here
    
    #define your features
    features = [1, 4] #change this if needed
    
    #select the data using features
    train_data_matrix = train_data[:,features,:]
    test_data_matrix = test_data[:,features]
    
    #define the shape of your label matrix
    pred_labels_kmeans = np.empty([train_data_matrix.shape[0],train_data_matrix.shape[2]])
    
    for i in range(train_data.shape[2]): #predict for each subject in the traindata the labels using kmeans
        #normalize data (only needed for kmeans, because normalizing data already happens in the ckNN and cAtlases function)
        train_data, test_data = seg.normalize_data(train_data_matrix[:,:, i], test_data_matrix)
        
        #find the optimized clusterpoints
        #1) kmeans will show start and end plot with the cluster centers
#        _, _, w_final = kmeans(train_data, train_labels[:,i], num_iter, mu) #realize train_labels is only needed for plotting
        #2) kmeans will not show any plots
        _, _, w_final = kmeans_no_plot(train_data, train_labels[:,i], num_iter, mu) #realize train_labels is only needed for plotting

        #predict the data 
        temp_pred = predicted_kmeans_test(w_final, test_data)
        
        #store the predicted lables for each subject
        pred_labels_kmeans[:,i] = temp_pred
        
    #decision fusion based on majority voting
    predicted_labels_kmeans_final = scipy.stats.mode(pred_labels_kmeans, axis = 1)[0].flatten()
    
    #get the labels from the other two methods: combined_atlas and combined knn
    _, pred_labels_cat = seg.segmentation_combined_atlas(train_labels, combining='mode') 
    _ ,pred_labels_cnn = seg.segmentation_combined_knn(train_data_matrix, train_labels, test_data_matrix, 1)
    
    #concatenate all predictions from the three 'submethods'
    pred_labels_cat = pred_labels_cat.T
    pred_labels_cnn = pred_labels_cnn.T
    concat_labels = np.vstack((predicted_labels_kmeans_final, pred_labels_cat, pred_labels_cnn)).T
    
    #decision fusion based on majority voting for the three submethods together
    predicted_labels = scipy.stats.mode(concat_labels, axis = 1)[0]
    #------------------------------------------------------------------#
    return predicted_labels


def segmentation_demo():
    #only SECTION 2 is needed for what we want to do
    train_subject = 1
    test_subject = 2
    train_slice = 1
    test_slice = 1
    task = 'tissue'
    #SECTION 1 (this seciton has nothing to do with SECTION 2) 
    
    #Load data from a train and testsubject
    train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
    test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)
    
    util.scatter_data(train_data, train_labels, 0, 6)
    util.scatter_data(test_data, test_labels, 0,6)
    
    predicted_labels = seg.segmentation_atlas(None, train_labels, None)

    err = util.classification_error(test_labels, predicted_labels)
    dice = util.dice_overlap(test_labels, predicted_labels)

    #Display results
    true_mask = test_labels.reshape(240, 240)
    predicted_mask = predicted_labels.reshape(240, 240)

    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(true_mask, 'gray')
    ax1.imshow(predicted_mask, 'viridis', alpha=0.5)
    print('Subject {}, slice {}.\nErr {}, dice {}'.format(test_subject, test_slice, err, dice))

    ## SECTION 2:Compare methods
    num_images = 5
    num_methods = 3
    im_size = [240, 240]

    all_errors = np.empty([num_images,num_methods])
    all_errors[:] = np.nan
    all_dice = np.empty([num_images,num_methods])
    all_dice[:] = np.nan

    all_subjects = np.arange(5) #list of all subjects [0, 1, 2, 3, 4]
    train_slice = 2
    task = 'tissue'
    all_data_matrix = np.empty([train_data.shape[0],train_data.shape[1],num_images])
    all_labels_matrix = np.empty([train_labels.size,num_images])

    #Load datasets once
    print('Loading data for ' + str(num_images) + ' subjects...')

    for i in all_subjects:
        sub = i+1
        train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
        all_data_matrix[:,:,i] = train_data
        all_labels_matrix[:,i] = train_labels.flatten()

    print('Finished loading data.\nStarting segmentation...')

    #Go through each subject, taking i-th subject as the test
    for i in all_subjects:
        sub = i+1
        #Define training subjects as all, except the test subject
        train_subjects = all_subjects.copy()
        train_subjects = np.delete(train_subjects, i)

        train_data_matrix = all_data_matrix[:,:,train_subjects]
        train_labels_matrix = all_labels_matrix[:,train_subjects]
        test_data = all_data_matrix[:,:,i]
        test_labels = all_labels_matrix[:,i]
        test_shape_1 = test_labels.reshape(im_size[0],im_size[1])

        fig = plt.figure(figsize=(15,5))

        predicted_labels, predicted_labels2 = seg.segmentation_combined_atlas(train_labels_matrix)
        all_errors[i,0] = util.classification_error(test_labels, predicted_labels2)
        all_dice[i,0] = util.dice_multiclass(test_labels, predicted_labels2)
        predicted_mask_1 = predicted_labels2.reshape(im_size[0],im_size[1])
        ax1 = fig.add_subplot(131)
        ax1.imshow(test_shape_1, 'gray')
        ax1.imshow(predicted_mask_1, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,0], all_dice[i,0])
        ax1.set_xlabel(text_str)
        ax1.set_title('Subject {}: Combined atlas'.format(sub))

        predicted_labels,predicted_labels2 = seg.segmentation_combined_knn(train_data_matrix,train_labels_matrix,test_data)
        all_errors[i,1] = util.classification_error(test_labels, predicted_labels2)
        all_dice[i,1] = util.dice_multiclass(test_labels, predicted_labels2)
        predicted_mask_2 = predicted_labels2.reshape(im_size[0],im_size[1])
        ax2 = fig.add_subplot(132)
        ax2.imshow(test_shape_1, 'gray')
        ax2.imshow(predicted_mask_2, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,1], all_dice[i,1])
        ax2.set_xlabel(text_str)
        ax2.set_title('Subject {}: Combined k-NN'.format(sub))

        #OUR METHOD
        #predict the labels using our method
        predicted_labels_mymethod = segmentation_mymethod(train_data_matrix,train_labels_matrix,test_data,num_iter = 100,mu = 0.1)
        
        #determine error and dice (multiclass, since there are more classes)
        all_errors[i,2] = util.classification_error(test_labels, predicted_labels_mymethod)
        all_dice[i,2] = util.dice_multiclass(test_labels, predicted_labels_mymethod)
        
        #reshape the predicted labels in order to plot the results
        predicted_mask_3 = predicted_labels_mymethod.reshape(im_size[0],im_size[1])
        
        #plot the predicted image over the real image
        plt.imshow(predicted_mask_3, 'viridis')
        ax3 = fig.add_subplot(133)
        ax3.imshow(test_shape_1, 'gray')
        ax3.imshow(predicted_mask_3, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,2], all_dice[i,2])
        ax3.set_xlabel(text_str)
        ax3.set_title('Subject {}: My method'.format(sub))
        
        #save the figure after every loop (3 subimages/plots)
        fig.savefig("Results for test subject {}".format(sub), )

def funX(X):
    return lambda w: seg.cost_kmeans(X,w)

def kmeans(X, labels, num_iter, mu = 0.1):
#    X,_ = seg.normalize_data(X)
    N, M = X.shape
    #Define number of clusters we want
    clusters = 4;

    # Cost function used by k-Means
    # fun = lambda w: seg.cost_kmeans(X,w)
    fun = funX(X)

    ## Algorithm
    #Initialize cluster centers
    idx = np.random.randint(N, size=clusters)
    initial_w = X[idx,:]
    w_draw = initial_w
    print(w_draw)

    #Reshape into vector (needed by ngradient)
    w_vector = initial_w.reshape(clusters*M, 1)

    #Vector to store cost
    xx = np.linspace(1, num_iter, num_iter)
    kmeans_cost = np.empty(*xx.shape)
    kmeans_cost[:] = np.nan

    fig = plt.figure(figsize=(14,6))
    ax1  = fig.add_subplot(121)
    util.scatter_data(X,labels,ax=ax1)

    line1, = ax1.plot(w_draw[:,0], w_draw[:,1], "k*",markersize=10, label='W-vector')
    # im3  = ax1.scatter(w_draw[:,0], w_draw[:,1])
    ax1.grid()

    ax2  = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 10))

    text_str = 'k={}, g={:.2f}\ncost={:.2f}'.format(0, 0, 0)

    txt2 = ax2.text(0.3, 0.95, text_str, bbox={'facecolor': 'green', 'alpha': 0.4, 'pad': 10},
             transform=ax2.transAxes)

#     xx = xx.reshape(1,-1)
    line2, = ax2.plot(xx, kmeans_cost, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.grid()

    for k in np.arange(num_iter):

        # gradient ascent
        g = util.ngradient(fun,w_vector)
        w_vector = w_vector - mu*g.T
        # calculate cost for plotting
        kmeans_cost[k] = fun(w_vector)
        text_str = 'k={}, cost={:.2f}'.format(k, kmeans_cost[k])
        txt2.set_text(text_str)
        # plot
        line2.set_ydata(kmeans_cost)
        w_draw_new = w_vector.reshape(clusters, M)
        line1.set_data(w_draw_new[:,0], w_draw_new[:,1])
#        display(fig)
        clear_output(wait = True)
        plt.pause(.005)
    display(fig)
    # TODO: Find distance of each point to each cluster center
    # Then find the minimum distances min_dist and indices min_index
    w_final = w_vector.reshape(clusters, M)

    D = scipy.spatial.distance.cdist(X, w_final, metric='euclidean') #distances between X and C
    min_index = np.argmin(D, axis=1)
    min_dist = np.zeros((len(min_index),1))
    for i in range(len(min_index)):
        min_dist[i,0] = D.item((i, min_index[i]))
     # Sort by intensity of cluster center
    sorted_order = np.argsort(w_final[:,0], axis=0)
    
    # Update the cluster indices based on the sorted order and return results in
    # predicted_labels
    predicted_labels = np.empty(*min_index.shape)
    predicted_labels[:] = np.nan
    
    for i in np.arange(len(sorted_order)):
        predicted_labels[min_index==sorted_order[i]] = i
    
    return kmeans_cost, predicted_labels, w_final

def kmeans_no_plot(X, labels, num_iter, mu = 0.1):
#    X,_ = seg.normalize_data(X)
    N, M = X.shape
    #Define number of clusters we want
    clusters = 4;

    # Cost function used by k-Means
    # fun = lambda w: seg.cost_kmeans(X,w)
    fun = funX(X)

    ## Algorithm
    #Initialize cluster centers
    idx = np.random.randint(N, size=clusters)
    initial_w = X[idx,:]
    w_draw = initial_w
    print(w_draw)

    #Reshape into vector (needed by ngradient)
    w_vector = initial_w.reshape(clusters*M, 1)

    #Vector to store cost
    xx = np.linspace(1, num_iter, num_iter)
    kmeans_cost = np.empty(*xx.shape)
    kmeans_cost[:] = np.nan

#    fig = plt.figure(figsize=(14,6))
#    ax1  = fig.add_subplot(121)
#    util.scatter_data(X,labels,ax=ax1)
#
#    line1, = ax1.plot(w_draw[:,0], w_draw[:,1], "k*",markersize=10, label='W-vector')
#    # im3  = ax1.scatter(w_draw[:,0], w_draw[:,1])
#    ax1.grid()
#
#    ax2  = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 10))
#
#    text_str = 'k={}, g={:.2f}\ncost={:.2f}'.format(0, 0, 0)
#
#    txt2 = ax2.text(0.3, 0.95, text_str, bbox={'facecolor': 'green', 'alpha': 0.4, 'pad': 10},
#             transform=ax2.transAxes)
#
##     xx = xx.reshape(1,-1)
#    line2, = ax2.plot(xx, kmeans_cost, lw=2)
#    ax2.set_xlabel('Iteration')
#    ax2.set_ylabel('Cost')
#    ax2.grid()

    for k in np.arange(num_iter):

        # gradient ascent
        g = util.ngradient(fun,w_vector)
        w_vector = w_vector - mu*g.T
        # calculate cost for plotting
        kmeans_cost[k] = fun(w_vector)
#        text_str = 'k={}, cost={:.2f}'.format(k, kmeans_cost[k])
#        txt2.set_text(text_str)
        # plot
#        line2.set_ydata(kmeans_cost)
#        w_draw_new = w_vector.reshape(clusters, M)
#        line1.set_data(w_draw_new[:,0], w_draw_new[:,1])
##        display(fig)
#        clear_output(wait = True)
#        plt.pause(.005)
#    display(fig)
    # TODO: Find distance of each point to each cluster center
    # Then find the minimum distances min_dist and indices min_index
    w_final = w_vector.reshape(clusters, M)

    D = scipy.spatial.distance.cdist(X, w_final, metric='euclidean') #distances between X and C
    min_index = np.argmin(D, axis=1)
    min_dist = np.zeros((len(min_index),1))
    for i in range(len(min_index)):
        min_dist[i,0] = D.item((i, min_index[i]))
     # Sort by intensity of cluster center
    sorted_order = np.argsort(w_final[:,0], axis=0)
    
    # Update the cluster indices based on the sorted order and return results in
    # predicted_labels
    predicted_labels = np.empty(*min_index.shape)
    predicted_labels[:] = np.nan
    
    for i in np.arange(len(sorted_order)):
        predicted_labels[min_index==sorted_order[i]] = i
    
    return kmeans_cost, predicted_labels, w_final


def predicted_kmeans_test(w_final, test_data):
    D = scipy.spatial.distance.cdist(test_data, w_final, metric='euclidean') #distances between X and C
    min_index = np.argmin(D, axis=1)
    min_dist = np.zeros((len(min_index),1))
    for i in range(len(min_index)):
        min_dist[i,0] = D.item((i, min_index[i]))
     # Sort by intensity of cluster center
    sorted_order = np.argsort(w_final[:,0], axis=0)
    
    # Update the cluster indices based on the sorted order and return results in
    # predicted_labels
    predicted_labels = np.empty(*min_index.shape)
    predicted_labels[:] = np.nan
    
    for i in np.arange(len(sorted_order)):
        predicted_labels[min_index==sorted_order[i]] = i
    return predicted_labels