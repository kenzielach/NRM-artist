import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from design_class import *
from scipy.optimize import curve_fit 

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

def add_hole_cent(rng, coord_options):
    a = np.array(rng.integers(low=0, high=len(coord_options)))
    hcoords = coord_options[a] # + [545, 545] # convert proposed hole center coords to coords in aperture array
    coord_options2 = np.delete(coord_options, a, 0)
    return np.array(hcoords), coord_options2

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
        #print('yay')
        return np.array(hcoords), hcoords_list, flag
    flag = 1
    return np.array(hcoords), hcoords_list #, flag

########################################################################################################
########################################################################################################
########################################################################################################

def check_redundancy(my_design, bw):
    """ Check mask baselines for redundancy

    Checks for any redundancy in the baselines of the proposed mask design. If redundancy is above 0%, reject the mask design.

    Args:
        my_design (object): An instance of design class. The proposed mask to be tested.
        bw (float): Bandwidth of the instrument; default is to assume infinitely narrow band (optional)

    Returns:
        bool: Returns 1 if the mask has any redundancy, returns 0 if mask is fully non-redundant.
    """
    pmask = 25 # radius in pixels for 0.5m mask
    #pmask = 31 # radius in pixels for 0.65m mask
    #psc = 
    lam = 5.23*1e-6 # wavelength to use in meters; central wavelength of the band
    avg = 0
    count = 0
    fact = 2*np.pi*206265/lam
    uv_rad = my_design.hrad # need to change this
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
    else:
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

def make_design(nholes, hrad, ap, return_vcoords=False, bw=0): 
    """ Generates mask design

    Generates a single non-redundant aperture mask design using the user-inputted number of holes and hole radius.

    Args:
        nholes (int): Number of mask holes.
        hrad (float): Radius of projected holes in meters.
        return_vcoords (bool): If True will also return list of hole (x,y) coordinates.
    Returns:
        object: Instance of the design class containing a single non-redundant aperture mask design.
        array (optional): Array containing list of hole (x,y) coordinates.
        
    """

    while 1: # keep looping until we get a valid design
        coord_options = ap.hcmb_coords * 1.0
        my_design = design(nholes, hrad) # initialize design object
        rng = np.random.default_rng(seed=None) # set random number generator
        for i in range(nholes): # keep adding and checking a single hole until it's acceptable
            my_design.xy_coords_cm[i, :], coord_options = add_hole_cent(rng, coord_options)
        my_design.get_xy_m() # convert (x,y) coords in cm to m
        my_design.get_uvs() # calculate design uv coordinates
        rcheck = check_redundancy(my_design, bw)  # check design for redundancy
        if rcheck == 0: # if this statement is true, exit the loop and return our final design
            if return_vcoords == True:
                return my_design, coord_options
            else:
                return my_design
        #my_design.plot_uv()
        #plot_design(my_design, ap.file)
            
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

########################################################################################################
########################################################################################################
########################################################################################################

def save_design(mask_design):
    n = mask_design.nholes
    np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/' + str(n) + 'hole_xycoords.npy', mask_design.xy_coords_m)
    np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/' + str(n) + 'hole_uvcoords.npy', mask_design.uv_coords)

########################################################################################################
########################################################################################################
########################################################################################################

#def calc_uv_param(mask_design):
    #n = 100
    #coordsx, coordsy = np.meshgrid(np.linspace(0, 1000, int(np.sqrt(n))), np.linspace(0, 500, int(np.sqrt(n))))
    #coordsx2, coordsy2 = np.meshgrid(np.linspace(500, 1000, int(np.sqrt(n)/2)), np.linspace(0, 1000, int(np.sqrt(n)/2)))
    #coords = np.vstack([coordsx.ravel(), coordsy.ravel()]).T
    #coords2 = np.vstack([coordsx2.ravel(), coordsy2.ravel()]).T
    #xy = np.loadtxt('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/SPIE_2024/9hole_nirc2_xycoords_cent.txt')
    #coords = np.append(mask_design.uv_coords, -mask_design.uv_coords, axis=0)

    #plt.figure()
    #plt.scatter(coords[:, 0], coords[:, 1])
    #plt.show()

    #flatx = np.sort(coords[:, 0])
    #flaty = np.sort(coords[:, 1])

    #p, resid, n1, n2, n3  = np.polyfit(flatx, flaty, deg=1, full=True)
    #print(p[1])
    #print(resid) # / (len(flatx) - 2))

    #plt.figure()
    #plt.scatter(np.arange(-10, 10), np.arange(-10, 10))
    #plt.scatter(flatx, flaty)
    #plt.show()

    return(p[0], p[1], resid)

def plot_ps(mask_design):
    coords = np.append(mask_design.uv_coords, -mask_design.uv_coords, axis=0)
    #flatx = np.sort(coords[:, 0])
    #flaty = np.sort(coords[:, 1])
    #plt.figure()
    #plt.scatter(np.arange(-10, 10), np.arange(-10, 10))
    #plt.scatter(flatx, flaty)
    #plt.show()

    #xlocs = np.linspace(-10, 10, num_bins)
    #ylocs = np.linspace(-10, 10, num_bins)

    #plt.figure()
    #plt.scatter(coords[:, 0], coords[:, 1])
    #for i in range(len(xlocs)):
    #    plt.axvline(xlocs[i], -10, 10)
    #    plt.axhline(ylocs[i], -10, 10)
    #plt.show()

    dim = 200

    stuff = np.zeros([dim, dim])
    x = np.round(np.linspace(-10, 10, dim+1), 1)
    coords2 = np.round(coords, 1)

    for i in range(dim):
        for j in range(dim):
            for k in coords2:
                if ([x[i], x[j]] == k).all() == 1:
                    stuff[i, j] = 1

    #plt.figure()
    #plt.imshow(stuff)
    #plt.show()

    #params, covar, fit, stdx, stdy = fit_ps_fft(stuff)
    #dim = int(np.sqrt(len(fit)))
    #print(np.abs(stdx - stdy))
    #plt.figure()
    #plt.imshow(fit.reshape(dim, dim))   
    #plt.colorbar()
    #plt.show()
    return stuff

def fit_ps_fft(stuff):
    fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.pad(stuff, 500))))
    FT = np.real(fft.conj()*fft)

    cent = np.real(FT[590:610, 590:610])
    #liney = cent[25, :] / np.max(cent[25, :])
    #linex = cent[:, 25] / np.max(cent[:, 25])
    #linexy = np.zeros(len(liney))
    #for i in range(len(linexy)):
        #linexy[i] = cent[i, i]
    #linexy = linexy / np.max(linexy)
    #linemean = (liney + linex + linexy) / 3

    #plt.figure()
    #plt.title('zoomed in FFT')
    #plt.imshow(cent)
    #plt.colorbar()
    #plt.show()

    #plt.figure()
    #plt.title('slices of zoomed FFT')
    #plt.scatter(range(len(liney)), liney)
    #plt.scatter(range(len(linex)), linex)
    #plt.scatter(range(len(linexy)), linexy)
    #plt.scatter(range(len(linexy)), linemean)
    #plt.legend(['mean'])
    #plt.show()

    xvals = np.linspace(-len(cent) / 2, len(cent) / 2, len(cent))
    #yvals = linemean
    def rotgauss2d(xy, A, B, D, E): 
        angle = E
        x = xy[0] * np.cos(angle) - xy[1] * np.sin(angle)
        y = xy[1] * np.cos(angle) + xy[0] * np.sin(angle)
        z = A*np.exp(-(1/B)*x**2 - (1/D)*y**2)
        return z.ravel()
    def sinc2(x, A, B, C):
        y = (A*np.sin(B*x + C) / x)**2
        return y
    
    xy = np.meshgrid(xvals, xvals)
    parameters, covariance = curve_fit(rotgauss2d, xy, cent.ravel(), maxfev=5000, bounds=((-1e5, 0, 0, 0), (1e3, 1e3, 1e3, 2*np.pi)))     
    if (parameters[1] or parameters[2]) <= 0:
        raise Exception('Warning: returned invalid fit')
    fit = rotgauss2d(xy, parameters[0], parameters[1], parameters[2], parameters[3])
    fit = np.max(cent.ravel()) * fit / np.max(fit)
    fit_unrot = rotgauss2d(xy, parameters[0], parameters[1], parameters[2], 0)
    fit_unrot = np.max(cent.ravel()) * fit_unrot / np.max(fit_unrot)
    ar = np.abs((parameters[1] / parameters[2]) - 1)
    std1 = np.std(fit_unrot.reshape([len(cent), len(cent)])[:, int(len(cent)/2)])
    std2 = np.std(fit_unrot.reshape([len(cent), len(cent)])[int(len(cent)/2), :])
    sum2 = parameters[1]**2 + parameters[2]**2
    #stdx = np.std(fit.reshape([len(cent), len(cent)])[:, int(len(cent)/2)])
    #stdy = np.std(fit.reshape([len(cent), len(cent)])[int(len(cent)/2), :])

    #print(parameters)

    #plt.figure()
    #plt.scatter(xvals, fit)
    #plt.title('best gaussian fit')
    #plt.scatter(range(len(linexy)), linemean)
    #plt.scatter(xvals, yvals)
    #plt.legend(['fit', 'data'])
    #plt.show()

    return parameters, covariance, fit, std1, std2, ar, cent, sum2

    
########################################################################################################
########################################################################################################
########################################################################################################
    
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
