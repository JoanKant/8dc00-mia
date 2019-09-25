# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:05:19 2019

@author: 20171880

parallelizing
"""

import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output
import registration_adapted_functions as reg_adapt

import registration_project as proj
#Select images: T1 and T1 moving and T2 slice
#paths of the images
I1_path = '../data/image_data/1_1_t1.tif'
Im1_path = '../data/image_data/1_1_t1_d.tif'
I2_path = '../data/image_data/1_1_t2.tif'
pool = mp.Pool(mp.cpu_count())

mu_list =np.arange(0.00001, 0.0001, 0.0005)

results = [pool.apply(proj.intensity_based_registration_affine_MI_adapted(I1_path,I2_path, mu)) for mu in mu_list] 

pool.close()