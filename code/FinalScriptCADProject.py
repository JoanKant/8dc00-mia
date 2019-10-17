# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:56:31 2019

@author: 20171880
"""
import cad_project as prj
# First part
E_test, E_test_small = prj.nuclei_measurement(batch_size = 1000)

#Second part
predictedY_test, E_test  = prj.nuclei_classification(mu, batch_size, num_iterations); 

#Third part

predictedY_test_auto, E_test_auto = prj.auto_nuclei_classification(mu, batch_size)