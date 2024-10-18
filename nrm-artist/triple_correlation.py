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
    fact = 1090/len(mask_tc)
    start = int(len(mask_tc)/2-1090/2)
    stop = int(len(mask_tc)/2+1090/2)
    mask_tc_subframe = scipy.ndimage.rotate(mask_tc[start:stop, start:stop], angle=180, reshape=False, order=3) 
    return mask_tc_subframe

def make_mask_from_coords(coords, hrad=0.01):
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
    #plt.title('Mask Aperture')
    plt.colorbar()
    plt.show()

main_dir = '/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/'
keck = aperture('keck')
coords = np.loadtxt(main_dir + 'masks/SPIE_2024/final_mask_files/unrotated/m/9hole1_hp_m.txt')
mask = make_mask_from_coords(coords)
print('finished making mask, working on tc...')
mask_tc = tc(mask)
mask_tc_subframe = subframe_tc(mask_tc)
plot_mask(mask_tc_subframe*keck.file)