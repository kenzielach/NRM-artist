import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from design_class import *

def check_placement(hcoords, hrad, rng, aperture):
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
                    print('Oh no! Bad hole placement :( Trying again!')
                    return False
    print('Yay! found a good hole placement.')
    return True

def check_hole_cent(hcoords, aperture):
    hy = hcoords[0]
    hx = hcoords[1]
    if aperture[hy, hx] == 0:
        return False
    else:
        return True

def add_hole(hrad, rng, aperture):
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
        if check_placement(hcoords, hrad, rng, aperture) == True:
            return np.array(coords)

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
                if red > 0:
                    return 1
    return 0

def plot_design(my_design, aperture):
    """ Plots finished design

    Generates a matplotlib plot of the Keck aperture with the finished mask design projected.

    Args:
        my_design (object): Instance of design class representing the finished mask design.
        aperture (array): Numpy array generated from the Keck aperture file provided.
    
    """

    hcoords = my_design.xy_coords + [545, 545] # convert hole center coords to coords in aperture array
    for i in range(1090):
        for j in range(1090):
            for a in range(my_design.nholes):
                if np.sqrt((i - hcoords[a, 0])**2 + (j - hcoords[a, 1])**2) < (100 * my_design.hrad):
    
                     aperture[i, j] = 0
    plt.figure()
    plt.imshow(aperture)
    plt.colorbar()
    plt.show()

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
        rng = np.random.default_rng(seed=None) # set random number generator
        aperture = pyfits.getdata('/Users/kenzie/Desktop/CodeAstro/planet-guts/keck_aperture.fits') # set Keck primary aperture
        for i in range(nholes): # keep adding and checking a single hole until it's acceptable
            my_design.xy_coords[i, :] = add_hole(hrad, rng, aperture)

        my_design.get_uvs() # calculate design uv coordinates

        rcheck = check_redundancy(my_design)  # check design for redundancy
        if rcheck == 1: # if true, there's some redundancy and we need to start over
            print("Uh-oh, mask has redundancies! Trying again...")
        if rcheck == 0: # if this statement is true, exit the loop and return our final design!
            print("Yay! Mask design is non-redundant. Plotting design...")
            #print(my_design.xy_coords)
            plot_design(my_design, aperture)
            return my_design