import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output
import registration_util as util
from cpselect.cpselect import cpselect
import registration_util as util
import numpy as np
import math


def point_based_registration():
    #paths of the images T1 slices
    I1_path = '../data/image_data/3_2_t1.tif'
    Im1_path = '../data/image_data/3_2_t1_d.tif'
    
    #paths of the images T1 and T2 slices
    I2_path = '../data/image_data/3_2_t1.tif'
    Im2_path = '../data/image_data/3_2_t2.tif'
    
    #read/load the images T1 image slices
    imageX1 = plt.imread(I1_path);
    imageX2 = plt.imread(Im1_path);
    
    #read/load the images T1 and T2
    imageX3 = plt.imread(I2_path);
    imageX4 = plt.imread(Im2_path)
    
    #padding needed for homogenous transformation matrix
    padding = np.array([[0, 0 ,1]])
    
    #select the fiducials using cp select (corresponding points)  
    X1, Xm1 = util.my_cpselect(I1_path, Im1_path)
    X2, Xm2 = util.my_cpselect(I2_path, Im2_path)
    
    #convert to homogenous coordinates using c2h
    X1_h = util.c2h(X1)
    Xm1_h= util.c2h(Xm1)    
    X2_h = util.c2h(X2)
    Xm2_h = util.c2h(Xm2)
    
    #compute affine transformation and make a homogenous transformation matrix
    T1 = reg.ls_affine(X1_h,Xm1_h)
    T1_padded = np.vstack([T1, padding])
    
    T2 = reg.ls_affine(X2_h, Xm2_h)
    T2_padded = np.vstack([T2, padding])
    
    #transfrom the moving image using the transformation matrix
    It1, Xt1 = reg.image_transform(imageX2, T1_padded)
    It2, Xt2 = reg.image_transform(imageX4, T2_padded)
    
    #display the results (from left to right: fixed image, moving image and applied transformation matrix to moving image)
    fig = plt.figure(figsize = (20,30))
    
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(imageX1) #plot fixed image
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(imageX2) #plot moving image
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(It1) #plot transformed moving image
    
    fig2 = plt.figure(figsize = (20,30))
    ax4 = fig2.add_subplot(131)
    im4 = ax4.imshow(imageX3)
    ax5 = fig2.add_subplot(132)
    im5 = ax5.imshow(imageX4)
    ax6 = fig2.add_subplot(133)
    im6 = ax6.imshow(It2)
    
    ax1.set_title('Fixed image')
    ax2.set_title('Moving image')
    ax3.set_title('Transformed moving image')
    
    ax4.set_title('Fixed image')
    ax5.set_title('Moving image')
    ax6.set_title('Transformed moving image')
    
    return  T1_padded, T2_padded
def Evaluate_point_based_registration(T1_padded, T2_padded):
     #paths of the images T1 slices
    I1_path = '../data/image_data/3_2_t1.tif'
    Im1_path = '../data/image_data/3_2_t1_d.tif'
    
    #paths of the images T1 and T2 slices
    I2_path = '../data/image_data/3_2_t1.tif'
    Im2_path = '../data/image_data/3_2_t2.tif'
    
    #read/load the images T1 image slices
    imageX1 = plt.imread(I1_path);
    imageX2 = plt.imread(Im1_path);
    
    #read/load the images T1 and T2
    imageX3 = plt.imread(I2_path);
    imageX4 = plt.imread(Im2_path)
    
    
    #select corresponding point pairs using cp select (corresponding points)  
    X1_target, Xm1_target = util.my_cpselect(I1_path, Im1_path)
    X2_target, Xm2_target = util.my_cpselect(I2_path, Im2_path)
       
    #transform the selected points
    Points1_target, Xt1= reg.image_transform(Xm1_target, T1_padded)
    Points2_target, Xt2= reg.image_transform(Xm2_target, T2_padded)
    
    #average distance between points (= target registration error)
    Av_dist1 = calculateAvg_Distance(X1_target, Points1_target)
    Av_dist2 = calculateAvg_Distance(X2_target, Points2_target)
    return Av_dist1, Av_dist2

def calculateAvg_Distance(points, points_t):  
    totaldistance = 0
    numberOfPoints = points.shape[1]
    for i in range(numberOfPoints):
         dist = math.sqrt((points_t[0,i] - points[0,i])**2 + (points_t[1,i] - points[1,i])**2)  
         totaldistance = totaldistance+dist
    average_distance = totaldistance/numberOfPoints        
    return average_distance

    
    
    
    
    
    
   