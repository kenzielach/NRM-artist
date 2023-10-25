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

def add_hole_cent(rng, hcoords_list, vcoords):
    a = np.array(rng.integers(low=0, high=len(vcoords)))
    hcoords = vcoords[a] # + [545, 545] # convert proposed hole center coords to coords in aperture array
    vcoords2 = np.delete(vcoords, a, 0)
    hcoords_list.append(hcoords)
    return np.array(hcoords), hcoords_list, vcoords2

########################################################################################################
########################################################################################################
########################################################################################################

def add_hole(rng, aperture, my_design, hcoords_list):
    """ Propose a new mask hole

    Generates a proposed hole (x,y) coordinate set, then calls check_placement() to check whether these coordinates are acceptable. If they are, then these coordinates are returned. 
    
    Args:
        hrad (float): Hole radius in meters.
        rng (object): Random number generator.
        aperture (array): numpy array containing a boolean mask of the Keck primary.
    
    Returns:
        array: Returns a numpy array of the accepted hole (x,y) coordinates.
    """
    i = 0
    flag = 0
    while i < 50:
        coords = np.array(rng.integers(low=-545, high=545, size=2))
        hcoords = coords + [545, 545] # convert proposed hole center coords to coords in aperture array
        if check_hole_cent(hcoords, aperture) == False:
            continue
        if check_spiders_gaps(hcoords, my_design.hrad, aperture) == False:
            continue
        if check_hole_overlap(hcoords, hcoords_list, my_design.hrad) == False:
            continue
        #if len(hcoords_list) > 0:
        #    tmp_xycoords = np.array(list(my_design.xy_coords_cm) + hcoords)
        #    tmp_design = design(my_design.nholes, my_design.hrad)
        #    tmp_design.xy_coords_cm = tmp_xycoords
        #    tmp_design.get_xy_m() # convert (x,y) coords in cm to m
        #    tmp_design.get_uvs()
        #    if check_redundancy(tmp_design) == 1:
        #        i += 1
        #        continue
        hcoords_list.append(hcoords)
        print('yay')
        return np.array(hcoords), hcoords_list, flag
    flag = 1
    return np.array(hcoords), hcoords_list, flag

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
    pmask = 25 # radius in pixels
    #psc = 
    avg = 0
    count = 0
    uv_rad = my_design.hrad
    n = 50000
    ci = 0
    for i in my_design.uv_coords:
        cj = 0
        b1 = i
        for j in my_design.uv_coords:
            b2 = j
            if ci == cj:
                cj += 1
                continue
            cj += 1
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
                avg += red
            count += 1
        ci += 1
    if count == 0:
        avg_f = 0
    else:
        avg_f = avg / count
    if avg_f > 0:
        return 1
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
    plt.show()

########################################################################################################
########################################################################################################
########################################################################################################

def make_design(nholes, hrad, return_vcoords=False): 
    """ Generates mask design

    Generates a single non-redundant aperture mask design using the user-inputted number of holes and hole radius.

    Args:
        nholes (int): Number of mask holes.
        hrad (float): Radius of projected holes in meters.

    Returns:
        object: Instance of the design class containing a single non-redundant aperture mask design.
    
    """

    vcoords0 = np.loadtxt('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/vcoords_keck9hole.txt')
    aperture = pyfits.getdata('/Users/kenzie/Desktop/CodeAstro/planet-guts/keck_aperture.fits') # set Keck primary aperture

    while 1: # keep looping until we get a valid design
        vcoords = vcoords0
        my_design = design(nholes, hrad) # initialize design object
        hcoords_list = []
        rng = np.random.default_rng(seed=None) # set random number generator
        for i in range(nholes): # keep adding and checking a single hole until it's acceptable
            my_design.xy_coords_cm[i, :], hcoords_list, vcoords = add_hole_cent(rng, hcoords_list, vcoords)
            #my_design.xy_coords_cm[i, :], hcoords_list, flag = add_hole(rng, aperture, my_design, hcoords_list)
            #if flag == 1:
                #break
        #if flag == 1:
            #print('hit a snag, starting over...')
            #continue
        my_design.get_xy_m() # convert (x,y) coords in cm to m
        my_design.get_uvs() # calculate design uv coordinates

        #print('Found holes, checking redundancy...')

        rcheck = check_redundancy(my_design)  # check design for redundancy
        #if rcheck == 1: # if true, there's some redundancy and we need to start over
        #    print("Uh-oh, mask has redundancies! " + "some baselines are redundant. Trying to fix...")
        if rcheck == 0: # if this statement is true, exit the loop and return our final design!
            if return_vcoords == True:
                return my_design, vcoords
            else:
                print("Yay! Mask design is non-redundant. Plotting design...")
                plot_design(my_design, aperture)
                return my_design
            
########################################################################################################
########################################################################################################
########################################################################################################   
    
def add_to_design(mask_design, diff, vcoords0):
    aperture = pyfits.getdata('/Users/kenzie/Desktop/CodeAstro/planet-guts/keck_aperture.fits') # set Keck primary aperture
    vcoords = vcoords0
    hcoords_list = mask_design.xy_coords_cm.tolist()
    rng = np.random.default_rng(seed=None)
    for i in range(diff):
        temp_design = design(mask_design.nholes + 1, mask_design.hrad)
        count = 0
        while 1:
            new_hole, hcoords_list, vcoords = add_hole_cent(rng, hcoords_list, vcoords)
            temp_design.xy_coords_cm = np.array(hcoords_list)
            temp_design.xy_coords_cm = np.append(temp_design.xy_coords_cm, np.reshape(new_hole, [1,2]), axis=0)
            temp_design.get_xy_m() # convert (x,y) coords in cm to m
            temp_design.get_uvs() # calculate design uv coordinates
            rcheck = check_redundancy(temp_design)
            if rcheck == 1: # if the new hole makes it redundant:
                np.delete(temp_design.xy_coords_cm, temp_design.nholes-1, axis=0) # delete what I just added to temp
                hcoords_list = mask_design.xy_coords_cm.tolist() # return hcoords_list to normal
                vcoords = vcoords0 # return vcoords to normal
                count += 1
                if count == len(vcoords0 - mask_design.nholes):
                    print("Couldn't find a non-redundant design :(")
                    return mask_design
            if rcheck == 0: # if the new hole is non-redundant:
                mask_design.xy_coords_cm = np.append(mask_design.xy_coords_cm, np.reshape(new_hole, [1,2]), axis=0) # add new hole to design
                mask_design.nholes += 1 # increase nholes
                mask_design.xy_coords_m = mask_design.xy_coords_cm / 100
                break # break out of while loop, go on to next hole
    print("Yay! Mask design is non-redundant. Plotting design...")
    plot_design(mask_design, aperture)
    return mask_design # return updated design

########################################################################################################
########################################################################################################
########################################################################################################

def save_design(mask_design):
    n = 9
    np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/9hole' + str(n) + '_xycoords.npy', mask_design.xy_coords_m)
    np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/9hole' + str(n) + '_uvcoords.npy', mask_design.uv_coords)