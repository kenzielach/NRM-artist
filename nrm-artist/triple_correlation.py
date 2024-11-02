# take in mask coord file and hole radius, calculate and plot its triple correlation superimposed on Keck aperture

import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from aperture_class import *
from state_class import *
import scipy

########################################################################################################
########################################################################################################
########################################################################################################

def tc(mask):
    """Calculates the triple correlation between three 1D arrays."""
    corr_xy = correlate(mask, mask, mode='full')
    corr_xyz = correlate(corr_xy, mask, mode='full')
    return corr_xyz

########################################################################################################
########################################################################################################
########################################################################################################

def subframe_tc(tc):
    fact = 1090/len(tc)
    start = int(len(tc)/2-1090/2)
    stop = int(len(tc)/2+1090/2)
    mask_tc_subframe = scipy.ndimage.rotate(tc[start:stop, start:stop], angle=180, reshape=False, order=3) 
    return mask_tc_subframe

########################################################################################################
########################################################################################################
########################################################################################################

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

########################################################################################################
########################################################################################################
########################################################################################################

def plot_mask(mask):
    plt.figure()
    plt.imshow(mask)
    plt.colorbar()
    plt.show()

########################################################################################################
########################################################################################################
########################################################################################################

def find_zeros(TC, res):
    # finds zeros in the triple correlation, checks if coords are allowed, then returns zero coords
    nx, ny = np.nonzero(np.where(TC == 0, 1, 0))
    coords = np.array([nx, ny]) - res/2 # convert to cm
    return is_allowed(coords)

########################################################################################################
########################################################################################################
########################################################################################################

def is_allowed(coords, hcs):
    # checks if the chosen coordinate is excluded or not by adding each new coord to existing design and multiplying by exclusion matrix
    path = '/Users/kenzie/Desktop/NRM-artist/NRM-artist/nrm-artist/Keck_ex_mat.npy'
    ex_mat = np.load(path)
    inds = np.empty(len(coords[0]))
    for i in range(len(coords[0])):
        inds[i] = np.nonzero(np.where(hcs == coords[i], 1, 0))

########################################################################################################
########################################################################################################
########################################################################################################

def make_level_arr(nholes, level, hcmb_coords, mcoords, rng):
    hns = np.empty(nholes-level) # number of possible hole locations
    for i in range(nholes-level): # set a randomized array of potential holes
        tmp = rng.integers(low=0, high=len(hcmb_coords)) # fix this to reduce to only the holes allowed by TC
        if mcoords[tmp] == 1:
            hns[i] = tmp

########################################################################################################
########################################################################################################
########################################################################################################

def initiate_level(nholes, my_state, hcmb_coords, rng): # get ready to start/restart at a new level
    if len(my_state.level_arrs[my_state.level]) == 0: # if we haven't been at this level before:
        hns = make_level_arr(nholes, my_state.level, hcmb_coords, my_state.mcoords, rng)
        my_state.add_to_level_arrs(hns)
        attempt = my_state.level_attempts() # this should be 0
    else: # if we're re-trying this level:
        hns = my_state.level_arrs[my_state.level]
        attempt = my_state.level_attempts() # should not be 0
    return hns, attempt

########################################################################################################
########################################################################################################
########################################################################################################

def make_design_TC(nholes, hrad, ap, geometry='cent'):
    inds_list = np.empty(nholes)
    rng = np.random.default_rng(seed=None) # set random number generator
    if geometry == 'cent':
        my_state = state(nholes, len(ap.hcmb_coords))
        while my_state.level < nholes:
            hns, attempt = initiate_level(nholes, my_state, ap.hcmb_coords, rng)
            if len(hns) == 0: # if there aren't any possible holes:
                my_state.remove(hn)
                my_state.reset_level() # reset hns and attempts for this level
                my_state.down_level() # go down a level
                hns, attempt = initiate_level(nholes, my_state, ap.hcmb_coords, rng)
            for j in range(attempt, len(hns)): # we branch out with selecting holes for each level according to the order of hns
                # this way we don't try the same config twice
                hn = hns[j]
                my_state.add(hn) # add the hole
                if is_allowed(my_state.mcoords) == 1: # if the hole is allowed:
                    inds_list[my_state.level] = hn # records index for each hole numbered by level
                    my_state.up_level_attempts()
                    my_state.up_level()
                    break
                else:
                    my_state.remove(hn)
                    my_state.up_level_attempts()
            # if we go through them all with no viable options, reset this level and go down to next one
            my_state.reset_level()
            my_state.down_level()
            my_state.remove(inds_list[my_state.level])
    else:
        raise Exception('error: have not written this code yet')