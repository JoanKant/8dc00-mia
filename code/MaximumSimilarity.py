# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:01:09 2019

@author: 20171880

Getting the maximum similarity and its index
"""
#import some modules
import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output
import registration_util as util
import registration_adapted_functions as reg_adapt
import registration_project as proj

#path to directory of the results
path = "C:/Users/20171880/Desktop/8dc00-mia/results/Final results"
#name of file 
filename = "/final_project_affine_mutual_info_T1_and_T1m_sim0.00015.npy"

fullpath = path+filename
maximum_simil, index_value = proj.MaximumSimilarityValue(fullpath)

display("Maximum similarity is: "+str(maximum_simil[0]))
display("Iteration: "+str(index_value[0][0]+1))