import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output
import registration_util as util
from cpselect.cpselect import cpselect
import registration_util as util
import numpy as np
import matplotlib.image as mpimg 


def point_based_registration():
    I1_path = '../data/image_data/3_2_t1.tif'
    Im1_path = '../data/image_data/3_2_t1_d.tif'
    
    imgX1= mpimg.imread(I1_path)
    imgX1m = mpimg.imread(Im1_path)
    
    X1, Xm1 = util.my_cpselect(I1_path, Im1_path)
    #convert to homogenous coordinates
    X1_h = util.c2h(X1)
    
    Xm1_1h= util.c2h(Xm1)    
    #
    #
    T1 = reg.ls_affine(X1_h,Xm1_1h)
    T1 = util.c2h(T1)
    It1, Xt1 = reg.image_transform(imgX1m, T1)
    
#    fig = plt.figure(figsize=(12,5))
#
#    ax1 = fig.add_subplot(131)
#    im11 = ax1.imshow(X1)
#    im12 = ax1.imshow(It1, alpha=0.7)

