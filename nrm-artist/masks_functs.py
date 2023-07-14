import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from design_class import *

def add_check_hole(rng, my_design):
    # do a simple check to make sure the hole center is ok:
    check1 = True
    while check1 == True:
        check2 = False
        while check2 == False: 
            hcoords = np.array(rng.integers(low=-545, high=545, size=2))
            hy = int(hcoords[0])
            hx = int(hcoords[1])
            if my_design.mask[hy, hx] == 1:
                check2 = True
        # if this is ok we can check the rest of the hole:
        fg = np.array([[[y,x] for x in range(1090)] for y in range(1090)]) # (x,y) coords for each pixel in keck aperture
        dist = np.sqrt(np.sum((fg - [hy, hx])**2, axis=2)) # distance from hole center to each coordinate pair
        hcoords_all = np.array(np.where(dist <= my_design.hrad)).T # make an array of coordinates in the hole
        for i in hcoords_all:
            hy = i[0]
            hx = i[1]
            if my_design.mask[hy, hx] == 0:
                print('Oh no! hole is bad :( Trying again...')
                continue
        check1 = False
    print('Yay! Hole added to design.')
    my_design.mask = update_aperture(hcoords, my_design)
    my_design.xy_coords.append(hcoords)
    return my_design

def update_aperture(hcoords, my_design):
    hy = int(hcoords[0])
    hx = int(hcoords[1])
    fg = np.array([[[y,x] for x in range(1090)] for y in range(1090)]) # (x,y) coords for each pixel in keck aperture
    dist = np.sqrt(np.sum((fg - [hy, hx])**2, axis=2)) # distance from hole center to each coordinate pair
    my_design.mask[np.where(dist <= my_design.hrad)] = 0 
    
    return my_design.mask

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
    status = 0
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
                    status = 1
    return status

def plot_design(my_design, aperture):
    """ Plots finished design

    Generates a matplotlib plot of the Keck aperture with the finished mask design projected.

    Args:
        my_design (object): Instance of design class representing the finished mask design.
        aperture (array): Numpy array generated from the Keck aperture file provided.
    
    """

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
    status = 1
    while status == 1: # keep looping until we get a valid design
        my_design = design(nholes, 100*hrad) # initialize design object
        rng = np.random.default_rng(seed=None) # set random number generator
        aperture = pyfits.getdata('/Users/kenzie/Desktop/CodeAstro/planet-guts/keck_aperture.fits') # set Keck primary aperture
        my_design.mask = aperture
        holes_added = 0
        while holes_added < nholes: # keep adding and checking a single hole until it's acceptable
            my_design = add_check_hole(rng, my_design)
            holes_added += 1

        my_design.get_uvs() # calculate design uv coordinates

        print('Design made! Now checking for redundancy...')
        status = check_redundancy(my_design)  # check design for redundancy
        if status == 1: # if true, there's some redundancy and we need to start over
            print("Uh-oh, mask has redundancies! Trying again...")
        if status == 0: # if this statement is true, exit the loop and return our final design!
            print("Yay! Mask design is non-redundant. Plotting design...")
            plot_design(my_design, aperture)
            return my_design