# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:33:52 2019

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


#---------------------------------------------------------------------#
  # load the training, validation and testing datasets
fn1 = '../data/linreg_ex_test.txt'
fn2 = '../data/linreg_ex_train.txt'
fn3 = '../data/linreg_ex_validation.txt'
# shape (30,2) numpy array; x = column 0, y = column 1
test_data = np.loadtxt(fn1)
# shape (20,2) numpy array; x = column 0, y = column 1
train_data = np.loadtxt(fn2)
# shape (10,2) numpy array; x = column 0, y = column 1
validation_data = np.loadtxt(fn3)

# plot the training dataset
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(train_data[:,0], train_data[:,1], '*')
ax.grid()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Training data')

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
predictedY = validationones.dot(Theta)
predictedY_test = util.addones(testX).dot(Theta)
E_validation =np.sum(np.square(np.subtract(predictedY, validationY)))
E_test  =np.sum(np.square(np.subtract(predictedY_test, testY)))
#---------------------------------------------------------------------#