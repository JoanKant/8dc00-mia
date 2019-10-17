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
import scipy
## dataset preparation
fn = '../data/nuclei_data_classification.mat'
mat = scipy.io.loadmat(fn)

test_images = mat["test_images"] # (24, 24, 3, 20730)
test_y = mat["test_y"] # (20730, 1)
training_images = mat["training_images"] # (24, 24, 3, 14607)
training_y = mat["training_y"] # (14607, 1)
validation_images = mat["training_images"] # (24, 24, 3, 14607)
validation_y = mat["training_y"] # (14607, 1)

## dataset preparation
imageSize = training_images.shape
# every pixel is a feature so the number of features is:
# height x width x color channels
numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
training_x = training_images.reshape(numFeatures, training_images.shape[3]).T.astype(float)
validation_x = validation_images.reshape(numFeatures, validation_images.shape[3]).T.astype(float)
test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

# the training will progress much better if we
# normalize the features
meanTrain = np.mean(training_x, axis=0).reshape(1,-1)
stdTrain = np.std(training_x, axis=0).reshape(1,-1)

training_x = training_x - np.tile(meanTrain, (training_x.shape[0], 1))
training_x = training_x / np.tile(stdTrain, (training_x.shape[0], 1))

validation_x = validation_x - np.tile(meanTrain, (validation_x.shape[0], 1))
validation_x = validation_x / np.tile(stdTrain, (validation_x.shape[0], 1))

test_x = test_x - np.tile(meanTrain, (test_x.shape[0], 1))
test_x = test_x / np.tile(stdTrain, (test_x.shape[0], 1))

## training linear regression model
#-------------------------------------------------------------------#
# TODO: Select values for the learning rate (mu), batch size
# (batch_size) and number of iterations (num_iterations), as well as
# initial values for the model parameters (Theta) that will result in
# fast training of an accurate model for this classification problem.
a = 5
mu_init =10**-a;
mu = mu_init;

batch_size = 500
r,c = training_x.shape
Theta  = 0.02*np.random.rand(c+1, 1)
num_iterations = 300
#-------------------------------------------------------------------#

xx = np.arange(100000)
loss = np.empty(*xx.shape)
loss[:] = np.nan
validation_loss = np.empty(*xx.shape)
validation_loss[:] = np.nan
g = np.empty(*xx.shape)
g[:] = np.nan

fig = plt.figure(figsize=(8,8))
ax2 = fig.add_subplot(111)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss (average per sample)')
ax2.set_title('mu = '+str(mu))
h1, = ax2.plot(xx, loss, linewidth=2) #'Color', [0.0 0.2 0.6],
h2, = ax2.plot(xx, validation_loss, linewidth=2) #'Color', [0.8 0.2 0.8],
ax2.set_ylim(0, 0.7)
ax2.grid()

text_str2 = 'iter.: {}, loss: {:.3f}, val. loss: {:.3f}'.format(0, 0, 0)
txt2 = ax2.text(0.3, 0.95, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)

E_validation = 1; 
E_validation_new = 0; 
k = 0
counter = 0; 
stopnow = 0
normgradient = 1;
while normgradient>0.1 and stopnow<1:
    
    # pick a batch at random
    idx = np.random.randint(training_x.shape[0], size=batch_size)

    training_x_ones = util.addones(training_x[idx,:])
        
    validation_x_ones = util.addones(validation_x)

    # the loss function for this particular batch
    loss_fun = lambda Theta: cad.lr_nll(training_x_ones, training_y[idx], Theta)

    # gradient descent
    # instead of the numerical gradient, we compute the gradient with
    # the analytical expression, which is much faster
    Theta_new = Theta - mu*cad.lr_agrad(training_x_ones, training_y[idx], Theta).T

    loss[k] = loss_fun(Theta_new)/batch_size
    validation_loss[k] = cad.lr_nll(validation_x_ones, validation_y, Theta_new)/validation_x.shape[0]
    
    
    #distance to zero (0,0)
    normgradient = np.linalg.norm(validation_loss[k])

    
    # visualize the training
    ax2.set_xlim(0, k)
    ax2.set_title('mu = {:.2}'.format(mu))

    h1.set_ydata(loss)
    h2.set_ydata(validation_loss)
 
    text_str2 = 'iter.: {}, loss: {:.3f}, val. loss={:.3f} '.format(k, loss[k], validation_loss[k])
    txt2.set_text(text_str2)

    Theta = None
    Theta = np.array(Theta_new)
    Theta_new = None
    tmp = None

    display(fig)
    clear_output(wait = True)
    plt.pause(.005)
    
    check_val = validation_loss[k];
    check_val_back = validation_loss[k-10]
#    
#
    if counter == 100:
        mu = mu/10;
        counter = 0;
  
    
    if k>200: 
        if round(validation_loss[k],3) == round(validation_loss[k-50],3):
            stopnow = 1; 
            print("Equal losses")
    k+=1
    counter +=1
    

#    ---------------------------------------------------------------------#
#     TODO: Compute the error for the trained model.

