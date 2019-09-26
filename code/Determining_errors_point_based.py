# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:45:05 2019

@author: 20171880

Determining the errors for point based registration
"""
import numpy as np
from IPython.display import display, clear_output

import registration_project as proj
import registration_util as util
import matplotlib.pyplot as plt

#Things to change
numberofpoints = 4 #change to 4, 8 or 12
set_of_images = 2 # 1= T1 and T1m and 2 = T1 and T2
path = "C:/Users/20171880/Desktop/8dc00-mia/results/resultsTransformationMatrix" #change to your path to resultsTransformationMatrix


#DO NOT CHANGE FOLLOWING LINES
#Select images: T1 and T1 moving and T2 slice
#paths of the images
I1_path = '../data/image_data/1_1_t1.tif'
Im1_path = '../data/image_data/1_1_t1_d.tif'
I2_path = '../data/image_data/1_1_t2.tif'

#read the images
I1 = plt.imread(I1_path)
Im1 = plt.imread(Im1_path)
Im2 = plt.imread(I2_path)


                                                                
if set_of_images == 1:
    full_path = path+"/transformation_matrix_T1_and_T1_moving_"+str(numberofpoints)+".npy"
    X1_target, Xm1_target = util.my_cpselect(I1_path, Im1_path)

else:         
    full_path = path+"/transformation_matrix_T1_and_T2_"+str(numberofpoints)+".npy"
    X1_target, Xm1_target = util.my_cpselect(I1_path, I2_path)


transformation_matrix = np.load(full_path) 
                                                               
"""
Evaluation of point-based affine image registration
"""


Reg_error1 = proj.Evaluate_point_based_registration(transformation_matrix, X1_target, Xm1_target)
display("The registration error will be printed, please write the error down in Word")
if set_of_images ==1: 
    print('Registration error for pair of T1 and T1 moving image slices:\n{}'.format(Reg_error1))
else: 
    print('Registration error for pair of T1 and T2 image slices:\n{}'.format(Reg_error1))
