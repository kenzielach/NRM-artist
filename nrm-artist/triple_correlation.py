# take in mask coord file and hole radius, calculate and plot its triple correlation superimposed on Keck aperture

import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from aperture_class import *
import scipy

def tc(mask):
    """Calculates the triple correlation between three 1D arrays."""
    corr_xy = correlate(mask, mask, mode='full')
    corr_xyz = correlate(corr_xy, mask, mode='full')
    return corr_xyz

def subframe_tc(tc):
    fact = 1090/len(tc)
    start = int(len(tc)/2-1090/2)
    stop = int(len(tc)/2+1090/2)
    mask_tc_subframe = scipy.ndimage.rotate(tc[start:stop, start:stop], angle=180, reshape=False, order=3) 
    return mask_tc_subframe

def make_mask_from_coords(coords, hrad=0.5):
    res = 1090 # resolution of matrix, in units of cm of projected aperture
    mask = np.zeros([res, res])
    for a in range(len(coords)):
        x = coords[a, 0]
        y =  coords[a, 1]
        for i in range(res):
            for j in range(res):
                dist = np.sqrt(((i-res/2)-x*100)**2 + ((j-res/2)-y*100)**2) # distance from matrix point to mask coord, 
                if dist <= 100*hrad:
                    mask[i, j] = 1
    return mask

def plot_mask(mask):
    plt.figure()
    plt.imshow(mask)
    plt.colorbar()
    plt.show()

#def find_zeros(TC):
    # finds zeros in the triple correlation, returns list of zero coords


#def is_allowed(coords):
    # checks if the chosen coordinate is excluded or not
