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
I1_path = '../data/image_data/1_1_t1.tif'
Im1_path = '../data/image_data/1_1_t1_d.tif'
I2_path = '../data/image_data/1_1_t2.tif'
savepath = 'C:/Users/20171880/Desktop/8dc00-mia/results'

path = 'C:/Users/20171880/Desktop/8dc00-mia/results'



mu_max = 0.001
mu_min = 0.0001
runs = 3
stepsize = (0.0001-0.00001)/runs
looplist = np.arange(mu_max, mu_min, -stepsize)

similarity_matrix = np.zeros((runs, 200))
for i in range(runs+1):
    mu = looplist[i]
    display("This is for mu = " + str(mu) + " " + str(i+1) + " out of " + str(runs))
    
    #configure the savepaths for figure and similarity matrix
    savepath_fig = path+'/mu = ' + str(mu)+'with_T1_ and_T2.png'
    savepath_sim_matrix = path +'/T1_and_T2_sim'+str(mu)
    
    #calculating the similarities and returning the list and figure
    sim, fig = proj.intensity_based_registration_affine_MI_adapted(I1_path, I2_path, mu)
    
    similarity_matrix[i] = np.asarray(sim).transpose()
    
    #saving figure and similarities for the multiple runs
    plt.savefig(savepath_fig)
    np.save(savepath_sim_matrix, similarity_matrix) 
   
    
    
#single run
#mu = 1
#sim, fig = proj.intensity_based_registration_affine_MI_adapted(I1_path, I2_path, mu)
#plt.savefig(savepath +'/T1_and_T2_with_mu = '+str(mu)+'.png')
#
#with open(savepath +'/sim'+str(mu), 'wb') as f: 
#    pickle.dump(sim, f)
#