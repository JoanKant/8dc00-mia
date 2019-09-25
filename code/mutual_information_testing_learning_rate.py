# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:26:44 2019

@author: 20171880
"""
import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output
import registration_adapted_functions as reg_adapt
import registration_project as proj

#Change your paths
I1_path = '../data/image_data/1_1_t1.tif'
Im1_path = '../data/image_data/1_1_t1_d.tif'
I2_path = '../data/image_data/1_1_t2.tif'

#Path for saving 
path = 'C:/Users/20171880/Desktop/8dc00-mia/results'

"""
METHOD 1: Test multiple learning rates
"""
#Give maximal and minimum learning rate
mu_max = 0.001
mu_min = 0.0001

#Number of runs (how many learning rates to test)
runs = 3

#Do not change the following lines
stepsize = (mu_max-mu_min)/runs
looplist = np.arange(mu_max, mu_min, -stepsize)
similarity_matrix = np.zeros((runs, 200)) #every row consists of all the similarities for every iteration

for i in range(runs):
    mu = looplist[i]
    display("This is for mu = " + str(mu) + " " + str(i+1) + " out of " + str(runs))
    
    #configure the savepaths for figure and similarity matrix
    #Change following lines with the right names, only change the strings 'with_T1...' etc.
    savepath_fig = path+'/mu = ' + str(mu)+'with_T1_ and_T2.png'
    savepath_sim_matrix = path +'/T1_and_T2_sim'+str(mu)
    
    #calculating the similarities and returning the list and figure
    sim, fig = proj.intensity_based_registration_affine_MI_adapted(I1_path, I2_path, mu)
    #place the similarities in the similarity_matrix
    similarity_matrix[i] = np.asarray(sim).transpose()
    
    #saving figure and similarities for the multiple runs
    plt.savefig(savepath_fig)
    np.save(savepath_sim_matrix, similarity_matrix) 
   
    
"""
Method 2: Test a single learning rate
"""
#savepath_fig = path+'/mu = ' + str(mu)+'with_T1_ and_T2.png'
#savepath_sim_matrix = path +'/T1_and_T2_sim'+str(mu)
   
#single run
#mu = 1
#sim, fig = proj.intensity_based_registration_affine_MI_adapted(I1_path, I2_path, mu)
#plt.savefig(savepath_fig)
#np.save(savepath_sim_matrix, sim) 