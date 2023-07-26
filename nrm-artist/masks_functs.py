import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from design_class import *

########################################################################################################
########################################################################################################
########################################################################################################

def check_valid_nholes(nholes):
    if nholes < 3:
        raise AttributeError(f"Oops! Please enter at least 3 holes for your mask.")
    
def check_valid_hrad(hrad):
    if hrad < 0.01:
        raise AttributeError(f'Oops! Hole size is too small.')
    if hrad > 0.77:
        raise AttributeError(f'Oops! Hole size is too big.')

########################################################################################################
########################################################################################################
########################################################################################################

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

    for i in range(1090):
        for j in range(1090):
            if np.sqrt((i - hcoords[0])**2 + (j - hcoords[1])**2) < (100 * hrad):
                if aperture[i, j] == 0:
                    return False
    return True

########################################################################################################
########################################################################################################
########################################################################################################

def add_hole(hrad, rng, aperture, hcoords_list):
    """ Propose a new mask hole

    Generates a proposed hole (x,y) coordinate set, then calls check_placement() to check whether these coordinates are acceptable. If they are, then these coordinates are returned. 
    
    Args:
        hrad (float): Hole radius in meters.
        rng (object): Random number generator.
        aperture (array): numpy array containing a boolean mask of the Keck primary.
    
    Returns:
        array: Returns a numpy array of the accepted hole (x,y) coordinates.
    """

    print('Placing hole...')
    while 1:
        coords = np.array(rng.integers(low=-545, high=545, size=2))
        hcoords = coords + [545, 545] # convert proposed hole center coords to coords in aperture array
        if check_hole_cent(hcoords, aperture) == False:
            continue
        if check_spiders_gaps(hcoords, hrad, aperture) == False:
            continue
        if check_hole_overlap(hcoords, hcoords_list, hrad) == False:
            continue
        print('Yay! Acceptable hole placement found.')
        hcoords_list.append(hcoords)
        return np.array(hcoords)

########################################################################################################
########################################################################################################
########################################################################################################

def check_redundancy(my_design):
    """ Check mask baselines for redundancy

    Checks for any redundancy in the baselines of the proposed mask design. If redundancy is above 0%, reject the mask design.

    Args:
        my_design (object): An instance of design class. The proposed mask to be tested.

    Returns:
        bool: Returns 1 if the mask has any redundancy, returns 0 if mask is fully non-redundant.
    """

    uv_rad = my_design.hrad
    n = 50000
    for i in my_design.uv_coords:
        b1 = i
        for j in my_design.uv_coords:
            if i[0] == j[0] and i[1] == j[1]:
                continue
            b2 = j
            d1 = np.sqrt((b1[0] - b2[0])**2 + (b1[1] - b2[1])**2) # both are positive
            d2 = np.sqrt((-b1[0] - b2[0])**2 + (-b1[1] - b2[1])**2) # one is negative
            d = np.min([d1, d2])
            if d < 2*uv_rad:
                test_uvs = np.random.uniform(low=0, high=2*uv_rad, size=(2,n))
                dist_b1 = np.sqrt((test_uvs[0,:] - uv_rad)**2 + (test_uvs[1,:] - uv_rad)**2)
                dist_b2 = np.sqrt((test_uvs[0,:] - (uv_rad + d))**2 + (test_uvs[1,:] - (uv_rad))**2)
                count1 = (dist_b1 <= uv_rad).sum()
                count2 = 0
                for q in range(n):
                    if dist_b1[q] <= uv_rad and dist_b2[q] < uv_rad:
                        count2 += 1
                red = 100 * np.round(count2 / count1, 2)
                #rbl1_h1 = # hole 1 of one redundant baseline
                #rbl1_h2 = # hole 2 of the same redundant baseline
                if red > 0:
                    return 1#, rbl1_h1, rbl1_h2
    return 0

########################################################################################################
########################################################################################################
########################################################################################################

#def replace_hole(hrad, rng, aperture, hcoords_list):
    
    
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
    for i in range(1090):
        for j in range(1090):
            for a in range(my_design.nholes):
                if np.sqrt((i - hcoords[a, 0])**2 + (j - hcoords[a, 1])**2) < (100 * my_design.hrad):
    
                     aperture[i, j] = 0
    plt.figure()
    plt.imshow(aperture)
    plt.colorbar()
    plt.show()

########################################################################################################
########################################################################################################
########################################################################################################

def make_design(nholes, hrad): 
    """ Generates mask design

    Generates a single non-redundant aperture mask design using the user-inputted number of holes and hole radius.

    Args:
        nholes (int): Number of mask holes.
        hrad (float): Radius of projected holes in meters.

    Returns:
        object: Instance of the design class containing a single non-redundant aperture mask design.
    
    """
   
    while 1: # keep looping until we get a valid design
        my_design = design(nholes, hrad) # initialize design object
        hcoords_list = []
        rng = np.random.default_rng(seed=None) # set random number generator
        aperture = pyfits.getdata('/Users/kenzie/Desktop/CodeAstro/planet-guts/keck_aperture.fits') # set Keck primary aperture
        for i in range(nholes): # keep adding and checking a single hole until it's acceptable
            my_design.xy_coords_cm[i, :] = add_hole(hrad, rng, aperture, hcoords_list)

        my_design.get_xy_m() # convert (x,y) coords in cm to m
        my_design.get_uvs() # calculate design uv coordinates

        rcheck = check_redundancy(my_design)  # check design for redundancy
        if rcheck == 1: # if true, there's some redundancy and we need to start over
            print("Uh-oh, mask has redundancies! " + "some baselines are redundant. Trying to fix...")
        if rcheck == 0: # if this statement is true, exit the loop and return our final design!
            print("Yay! Mask design is non-redundant. Plotting design...")
            plot_design(my_design, aperture)
            return my_design