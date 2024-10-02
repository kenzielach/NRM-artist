import numpy as np
from aperture_class import *

def check_hole_cent(hcoords, aperture):
    hy = hcoords[0]
    hx = hcoords[1]
    if aperture[hy, hx] == 0:
        return False
    else:
        return True

########################################################################################################
########################################################################################################
########################################################################################################

def check_hole_overlap(hcoords, hcoords_list, hrad):
    if len(hcoords_list) == 0:
        return True
    else:
        for i in hcoords_list:
            dist = np.sqrt((i[0] - hcoords[0])**2 + (i[1] - hcoords[1])**2)
            if dist < 100 * hrad:
                return False
            else:
                continue


########################################################################################################
########################################################################################################
########################################################################################################

def check_spiders_gaps(hcoords, hrad, aperture):
    """ Check placement of hole

    Called by add_hole(). Checks that a proposed hole doesn't overlap other holes or spiders or mirror segment edges, and that it falls within the Keck aperture. If a hole does not meet requirements, hole is discarded and add_hole() is called again. Repeats until an acceptable hole location is found.

    Args:
        coords (array): numpy array containing the proposed (x,y) coordinates. 
        design (object): Mask design object.
        rng (object): Random number generator.
        aperture (array): numpy array containing a boolean mask of the Keck primary.
    
    Returns:
        array: returns the accepted (x,y) coordinates
    """

    xvals = np.arange(0, 1090, 1).reshape([1, 1090])
    yvals = np.flip(np.arange(0, 1090, 1).reshape([1090, 1]))
    distances = np.sqrt((xvals - hcoords[1])**2 + (yvals - hcoords[0])**2)
    distances[distances < 1.0] = 1000.
    distances[distances < (100 * hrad)] = -100.0
    if np.min(distances + 200*aperture) < -10.0:
        return False
    return True

########################################################################################################
########################################################################################################
########################################################################################################

def plot_design(my_design, aperture):
    """ Plots finished design

    Generates a matplotlib plot of the Keck aperture with the finished mask design projected.

    Args:
        my_design (object): Instance of design class representing the finished mask design.
        aperture (array): Numpy array generated from the Keck aperture file provided.
    
    """

    hcoords = my_design.xy_coords_cm
    aperture2 = aperture.file * 1.0

    if np.ndarray.flatten(hcoords > 545).any():
        print('warning: coords not centered! fixing...')
        hcoords -= [545, 545]
    
    for i in range(1090):
        for j in range(1090):
            for a in range(my_design.nholes):
                if np.sqrt(((i - 545) - hcoords[a, 0])**2 + ((j - 545) - hcoords[a, 1])**2) < (100 * my_design.hrad):
                     aperture2[i, j] = 0
    
    plt.figure()
    plt.imshow(aperture2)
    plt.xticks([])
    plt.yticks([])
    plt.show()

########################################################################################################
########################################################################################################
########################################################################################################