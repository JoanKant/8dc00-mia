import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output

from scipy import ndimage



I = plt.imread('../data/cameraman.tif')

N1 = np.random.randint(255, size=(512, 512))
N2 = np.random.randint(255, size=(512, 512))

# mutual information of an image with itself
p1 = reg.joint_histogram(I, I)
MI1 = reg.mutual_information_e(p1)
MI2 = reg.mutual_information(p1)
#assert abs(MI1-MI2) < 10e-3, "Mutual information function with entropy is incorrectly implemented (difference with reference implementation test)"
print(abs(MI1-MI2))
#print('Test successful!')