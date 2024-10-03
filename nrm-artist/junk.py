def calc_spacing(mask_design):

    num_bins = 15
    uvs = np.append(mask_design.uv_coords, -mask_design.uv_coords, axis=0)

    xpos0 = -10
    ypos0 = -10
    count = 0
    my_bins = np.zeros(num_bins)
    bin_size = int(20 / int(np.sqrt(num_bins)))

    for i in range(int(np.sqrt(num_bins))):
        xpos = -10 + (i + 1) * bin_size
        ypos0 = -10
        for j in range(int(np.sqrt(num_bins))):
            ypos = -10 + (j + 1) * bin_size
            for k in uvs:
                #print('is ' + str(k) + ' within ' + str(xpos0) + ' <= x < ' + str(xpos) + ' and ' + str(ypos0) + ' <= y < ' + str(ypos) + '?')
                if k[0] < xpos and k[0] >= xpos0 and k[1] < ypos and k[1] >= ypos0:
                    my_bins[count] += 1
            count += 1
            ypos0 = ypos
        xpos0 = xpos

    return(np.var(my_bins), num_bins)

########################################################################################################
########################################################################################################
########################################################################################################   
    
def add_to_design(mask_design, diff, vcoords0):
    #aperture = pyfits.getdata('/Users/kenzie/Desktop/CodeAstro/planet-guts/keck_aperture.fits') # set Keck primary aperture
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
    #print("Yay! Mask design is non-redundant. Plotting design...")
    #plot_design(mask_design, aperture)
    return mask_design # return updated design

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
        #print('yay')
        return np.array(hcoords), hcoords_list, flag
    flag = 1
    return np.array(hcoords), hcoords_list #, flag

########################################################################################################
########################################################################################################
#######################################################################################################