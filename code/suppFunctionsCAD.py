# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:49:32 2019

@author: 20171880
"""
from builtins import print
import numpy as np
import cad_util as util
import segmentation_util as seg_util
import registration as reg
import segmentation as seg
import matplotlib.pyplot as plt
import cad
from IPython.display import display, clear_output, HTML

def linear_regression(train_data, test_data):
#    # plot the training dataset
#    fig = plt.figure(figsize=(10,10))
#    ax = fig.add_subplot(111)
#    ax.plot(train_data[:,0], train_data[:,1], '*')
#    ax.grid()
#    ax.set_xlabel('x')
#    ax.set_ylabel('y')
#    ax.set_title('Training data')

    #---------------------------------------------------------------------#
    # TODO: Implement training of a linear regression model.
    # Here you should reuse ls_solve() from the registration mini-project.
    # The provided addones() function adds a column of all ones to a data
    # matrix X in a similar way to the c2h() function used in registration.

    trainX = train_data[:,0].reshape(-1,1)
    trainXones = util.addones(trainX)
    trainY = train_data[:,1].reshape(-1,1)
       
        
    Theta, _ = reg.ls_solve(trainXones, trainY) 
    print(Theta)
    #---------------------------------------------------------------------#

    fig1 = plt.figure(figsize=(10,10))
    ax1 = fig1.add_subplot(111)
    util.plot_regression(trainX, trainY, Theta, ax1)
    ax1.grid()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(('Original data', 'Regression curve', 'Predicted Data', 'Error'))
    ax1.set_title('Training set')

    testX = test_data[:,0].reshape(-1,1)
    testY = test_data[:,1].reshape(-1,1)

    fig2 = plt.figure(figsize=(10,10))
    ax2 = fig2.add_subplot(111)
    util.plot_regression(testX, testY, Theta, ax2)
    ax2.grid()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(('Original data', 'Regression curve', 'Predicted Data', 'Error'))
    ax2.set_title('Test set')

    #---------------------------------------------------------------------#
    # TODO: Compute the error for the trained model.
    predictedY_test = util.addones(testX).dot(Theta)
    E_test  =np.sum(np.square(np.subtract(predictedY_test, testY)))
    #---------------------------------------------------------------------#

    return E_test, predictedY_test