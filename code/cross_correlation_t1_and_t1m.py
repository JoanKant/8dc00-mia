# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 05:51:24 2019

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
path = 'C:/Users/20171880/Desktop/8dc00-mia/results'

#multiple run
numberofsteps = 1
min_value = 0.00005
max_value = 0.00001
stepsize = (max_value-min_value)/numberofsteps

mu = 0.01
similarity_matrix = np.zeros((2, 200))
for i in range(2):
    display("This is for mu = " + str(mu) + " " + str(i+1) + " out of " + str(2))
    
    #configure the savepaths for figure and similarity matrix
    savepath_fig = path+'/mu = ' + str(mu)+'with_T1_ and_T1m.jpg'
    savepath_sim_matrix = path +'/T1_and_T1_m_sim'+str(mu)
    
    #calculating the similarities and returning the list and figure
    sim, fig = proj.intensity_based_registration_rigid_Corr_adapted(I1_path, Im1_path, mu)
    
    similarity_matrix[i] = np.asarray(sim).transpose()
    
    #saving figure and similarities for the multiple runs
    plt.savefig(savepath_fig,format = 'jpg')
    np.save(savepath_sim_matrix, similarity_matrix) 
    mu/=10
    
    
#single run
#mu = 1
#sim, fig = proj.intensity_based_registration_affine_MI_adapted(I1_path, I2_path, mu)
#plt.savefig(savepath +'/T1_and_T2_with_mu = '+str(mu)+'.png')
#
#with open(savepath +'/sim'+str(mu), 'wb') as f: 
#    pickle.dump(sim, f)
#
#    

