# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:42:58 2019

@author: 20171880
"""
#importing needed libraries/files etc
import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output
import registration_util as util
import registration_adapted_functions as reg_adapt
import registration_project as proj


"""
Settings
"""
 
#Select images: T1 and T1 moving and T2 slice
#paths of the images
I1_path = '../data/image_data/1_1_t1.tif'
Im1_path = '../data/image_data/1_1_t1_d.tif'
I2_path = '../data/image_data/1_1_t2.tif'

#read the images
I1 = plt.imread(I1_path)
Im1 = plt.imread(Im1_path)
Im2 = plt.imread(I2_path)

path = 'C:/Users/20171880/Desktop/8dc00-mia/results/Point_based_registration'

"""
Part 1: T1 and T1 moving
"""

numberofpoints = 12

#choose fiducial points and save them for T1 and T1 moving
#display("Choose fiducial points for T1 and T2 moving")
#X1, Xm1 = util.my_cpselect(I1_path, Im1_path)
#
#
##determine the transformation matrix
#display("Determine the transformation matrix")
#Th1, fig1, It1 = proj.point_based_registration(I1_path,Im1_path, X1, Xm1)
#np.save(path+"/transformation_matrix_T1_and_T1_moving_"+str(numberofpoints), Th1)
#plt.savefig(path+"/transformed_T1_moving_image_"+str(numberofpoints)+".png")
#
#
##calculate correlation
#display("Calculate correlation")
#corr_I1 = reg.correlation(I1, It1)
#
##calculate mutual information
#display("Calculate mutual information")
#p1 = reg.joint_histogram(I1, It1)
#mi_I1 = reg.mutual_information(p1)




"""
Part 2: T1 and T2
""" 

##choose fiducial points and save them for T1 and T1 moving
X1, X2 = util.my_cpselect(I1_path, I2_path)

Th2,fig2,It2 = proj.point_based_registration(I1_path,I2_path, X1, X2)
np.save(path+"/transformation_matrix_T1_and_T2_moving_"+str(numberofpoints), Th2)
plt.savefig(path+"/transformed_T2_image_"+str(numberofpoints)+".png")

#calculate correlation
display("Calculate correlation")
corr_I2 = reg.correlation(I1, It2)
#calculate mutual information
display("Calculate mutual information")
p2 = reg.joint_histogram(I1, It2)
mi_I2 = reg.mutual_information(p2)


"""
Evaluation of point-based affine image registration
"""

#Calculate the registration error for T1 and T1 moving
#
##Select target points for T1 and T1 moving
#X1_target, Xm1_target = util.my_cpselect(I1_path, Im1_path)
#
#Reg_error1 = proj.Evaluate_point_based_registration(T1, X1_target, Xm1_target)
#print('Registration error for pair of T1 image slices:\n{}'.format(Reg_error1))
#
##Select target points for T1 and T2
#X1_target, X2_target = util.my_cpselect(I2_path, Im2_path)
##Calculate the registration error for T1 and T2
#Reg_error2 = project.Evaluate_point_based_registration(T2, X2_target, Xm2_target)
#print('Registration error for T1 and T2 image slice:\n{}'.format(Reg_error_2))