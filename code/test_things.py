import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output
from cpselect.cpselect import cpselect


from scipy import ndimage
I_path = '../data/image_data/1_1_t1.tif'
Im_path = '../data/image_data/1_1_t1_d.tif'



controlpointlist = cpselect(I_path, Im_path)

#number of points
numberOfPoints = len(controlpointlist)


X = np.zeros((2,numberOfPoints, 1))
Xm = np.zeros((2,numberOfPoints, 1))
for i in range(numberOfPoints):
    
    valuesOfPoint = [*controlpointlist[i].values()]
    X[0,i] = valuesOfPoint[1]
    X[1,i] = valuesOfPoint[2]
    
    Xm[0,i] = valuesOfPoint[3]
    Xm[1,i] = valuesOfPoint[4]



