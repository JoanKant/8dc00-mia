"""
Project code+scripts for 8DC00 course.
"""

import numpy as np
import cad_util as util
import matplotlib.pyplot as plt
import registration as reg
import cad
import scipy
from IPython.display import display, clear_output
import scipy.io
import suppFunctionsCAD as sup
import random
import segmentation_util as seg_util

def nuclei_measurement(batch_size = 1000):

    fn = '../data/nuclei_data.mat'
    mat = scipy.io.loadmat(fn)
    test_images = mat["test_images"] # shape (24, 24, 3, 20730)
    test_y = mat["test_y"] # shape (20730, 1)
    training_images = mat["training_images"] # shape (24, 24, 3, 21910)
    training_y = mat["training_y"] # shape (21910, 1)

    montage_n = 300
    sort_ix = np.argsort(training_y, axis=0)
    sort_ix_low = sort_ix[:montage_n] # get the 300 smallest
    sort_ix_high = sort_ix[-montage_n:] #Get the 300 largest

    # visualize the 300 smallest and the 300 largest nuclei
    X_small = training_images[:,:,:,sort_ix_low.ravel()]
    X_large = training_images[:,:,:,sort_ix_high.ravel()]
    fig = plt.figure(figsize=(16,8))
    ax1  = fig.add_subplot(121)
    ax2  = fig.add_subplot(122)
    util.montageRGB(X_small, ax1)
    ax1.set_title('300 smallest nuclei')
    util.montageRGB(X_large, ax2)
    ax2.set_title('300 largest nuclei')
    
    # dataset preparation
    imageSize = training_images.shape

    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
    print((numFeatures))
    training_x = training_images.reshape(numFeatures, imageSize[3]).T.astype(float)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

  
    #Predict Y (= area) for the test set and return the error too
    E_test, predicted_y = sup.linear_regression(training_x, test_x, numFeatures)

    # visualize the results
    fig2 = plt.figure(figsize=(16,8))
    ax1  = fig2.add_subplot(121)
    line1, = ax1.plot(predicted_y, test_y, ".g", markersize=3)
    ax1.grid()
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Predicted Area')
    ax1.set_title('Training with full sample')

    #training with smaller number of training samples
    #---------------------------------------------------------------------#
    # TODO: Train a model with reduced dataset size (e.g. every fourth
    # training sample).
    
    #Choose the samples randomly, only set the number of samples (original 21910 samples)
    ix = np.random.randint(imageSize[3], size=batch_size)
    
    #Select the train data (only select certain samples)
    training_x = training_images[:,:,:,ix].reshape(numFeatures, len(ix)).T.astype(float)
    
    E_test_small, predicted_y = sup.linear_regression(training_x, test_x, batch_size);
 
    #Evaluation 
    print("The error for testset using traindata consisting of all samples: {:.2f}".format(E_test))
    print("The error for testset using traindata consisting of less samples: {:.2f}".format(E_test_small))
    #---------------------------------------------------------------------#

    # visualize the results
    ax2  = fig2.add_subplot(122)
    line2, = ax2.plot(predicted_y, test_y, ".g", markersize=3)
    ax2.grid()
    ax2.set_xlabel('Area')
    ax2.set_ylabel('Predicted Area')
    ax2.set_title('Training with smaller sample')
    fig2.savefig("Predicted area and real area with batch size {}".format(batch_size)) 

    return E_test, E_test_small


#mu = 0.001, batch_size = 30, num_iterations = 200

def nuclei_classification(mu, batch_size, num_iterations):
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
#    mu = 0.00001
#    batch_size = 500
#    num_iterations = 300
    
    
    r,c = training_x.shape
   
    Theta  = 0.02*np.random.rand(c+1, 1)
        
 
        
    
    #-------------------------------------------------------------------#

    xx = np.arange(num_iterations)
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
    ax2.set_xlim(0, num_iterations)
    ax2.grid()

    text_str2 = 'iter.: {}, loss: {:.3f}, val. loss: {:.3f}'.format(0, 0, 0)
    txt2 = ax2.text(0.3, 0.95, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)

    for k in np.arange(num_iterations):
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

        # visualize the training
        h1.set_ydata(loss)
        h2.set_ydata(validation_loss)
        text_str2 = 'iter.: {}, loss: {:.3f}, val. loss={:.3f}'.format(k, loss[k], validation_loss[k])
        txt2.set_text(text_str2)

        Theta = None
        Theta = np.array(Theta_new)
        Theta_new = None
        tmp = None

        display(fig)
        clear_output(wait = True)
        plt.pause(.005)
    
    #save the final loss curve
    plt.savefig("Loss curve for batch size {} and init mu {:.2}.png".format(batch_size, mu)) 
  
 #   ---------------------------------------------------------------------#
#     TODO: Compute the error for the trained model.
    predictedY_test = util.addones(test_x).dot(Theta)
    E_test  =np.sum(np.square(np.subtract(predictedY_test, test_y)))
    return predictedY_test, E_test 

def auto_nuclei_classification(mu, batch_size):
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
#    a = 5
#    mu_init =10**-a;
#    mu = mu_init;
    
    #number of training samples
#    batch_size = 3000
    r,c = training_x.shape
    
    #initial weights 
    Theta  = 0.02*np.random.rand(c+1, 1)
    
    #-------------------------------------------------------------------#
    
    xx = np.arange(100000)
    loss = np.empty(*xx.shape)
    loss[:] = np.nan
    validation_loss = np.empty(*xx.shape)
    validation_loss[:] = np.nan
    g = np.empty(*xx.shape)
    g[:] = np.nan
    
    
    idx = np.random.randint(training_x.shape[0], size=batch_size)
    
    
    # Create base figure
    fig = plt.figure(figsize=(15,10))
  
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
    
    #Some initial parameter settings
    k = 0 #iteration number
    counter = 0; #count number of iterations, resets after 100 iterations
    stopnow = 0 #used to stop when loss doesn't decrease any further
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
        
        #Caclulate the loss and the validation loss
        loss[k] = loss_fun(Theta_new)/batch_size
        validation_loss[k] = cad.lr_nll(validation_x_ones, validation_y, Theta_new)/validation_x.shape[0]
    
        #distance to zero (0,0) used for minimizing loss
        normgradient = np.linalg.norm(validation_loss[k])    
        
        # visualize the training
        ax2.set_xlim(0, k) #axis needs to be adapted every iteration
        ax2.set_title('mu = {:.2}'.format(mu)) 
    
        h1.set_ydata(loss)
        h2.set_ydata(validation_loss)
     
        text_str2 = 'iter.: {}, loss: {:.3f}, val. loss={:.3f} '.format(k, loss[k], validation_loss[k])
        txt2.set_text(text_str2)
    



        display(fig)
        clear_output(wait = True)   
    
 
        #Set the new weights
        Theta = None
        Theta = np.array(Theta_new)
        Theta_new = None
        tmp = None
             
        display(fig)
        clear_output(wait = True)
        plt.pause(.005)
   
        #Stop when the validation_loss doesn't decrease any further by comparing the current validation loss with x iterations ago     
        if k>100: 
            if round(validation_loss[k],4) == round(validation_loss[k-25],4):
                stopnow = 1; 
                print("The validation loss has reached its equilibrium")
        
        #increment iteration parameters
        k+=1
        counter +=1
  
    #save the final loss curve
    fig.savefig("Loss curve for batch size {} and init mu {:.2}.png".format(batch_size, mu)) 
  
    #predict the test data with the final weights
    predictedY_test = util.addones(test_x).dot(Theta)
    
    #calculate the error for the test squared error
    E_test  =np.sum(np.square(np.subtract(predictedY_test, test_y)))
        
    return predictedY_test, E_test 
