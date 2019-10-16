# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:56:31 2019

@author: 20171880
"""
import cad_project as prj
#
##Linear regression for nuclei area measurement
#print("Linear regression for nuclei area measurement")
#prj.nuclei_measurement()
#
##Logistic regression for nuclei classification
#print("Logistic regression for nuclei classification")

#for i in range
predictedY_test, E_test  = prj.nuclei_classification(mu = 0.0001, batch_size = 30, num_iterations = 200); 
