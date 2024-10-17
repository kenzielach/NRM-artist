import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from design_class import *
from scipy.optimize import curve_fit 
from pupil_plane import *
from redundancy_check import *
from aperture_class import *

########################################################################################################
########################################################################################################
########################################################################################################

def pick_hole_cent(rng, coord_options):
    a = np.array(rng.integers(low=0, high=len(coord_options)))
    coord_options2 = np.delete(coord_options, a, 0)
    return coord_options[a], coord_options2

########################################################################################################
########################################################################################################
########################################################################################################

#def pick_hole(rng):


########################################################################################################
########################################################################################################
########################################################################################################

def start_from0(nholes, hrad, rng, bw, hcmb_coords, geometry='cent'):
            timeout = 0
            while 1:
                if timeout > 1000:
                    raise Exception(f'Timed out, could not find valid design.')
                my_design = design(nholes, hrad)
                coord_options = hcmb_coords * 1.0
                if geometry == 'cent':
                    coord_options = hcmb_coords * 1.0
                    for i in range(nholes):
                        hole, coord_options = pick_hole_cent(rng, coord_options)
                        my_design.add_hole(np.reshape(hole, [1, 2]), i)
                my_design.get_xy_m() # convert (x,y) coords in cm to m
                my_design.get_uvs() # calculate design uv coordinates
                rcheck = check_redundancy(my_design, bw)  # check design for redundancy
                if rcheck == 0: # if this statement is true, exit the loop and return our final design
                    return my_design
                timeout += 1

########################################################################################################
########################################################################################################
########################################################################################################

def start_from6(hcmb_coords, n, nholes, hrad, rng, bw, geometry='cent'):
    if geometry == 'cent':
        timeout1 = 0
        while 1:
            if timeout1 > 1000:
                raise Exception(f'Timed out, could not find a non-redundant design.')
            coord_options = hcmb_coords * 1.0
            my_design0 = design(n, hrad)
            for i in range(n):
                hole, coord_options = pick_hole_cent(rng, coord_options)
                my_design0.add_hole(np.reshape(hole, [1, 2]), i)
            my_design0.get_xy_m()
            my_design0.get_uvs()
            rcheck = check_redundancy(my_design0, bw)
            if rcheck == 0:
                final = add_to_6hole(n, nholes, hrad, rng, coord_options, bw, my_design0.xy_coords_cm, geometry)
                return final
            timeout1 += 1
    #if geometry == 'rand':

########################################################################################################
########################################################################################################
########################################################################################################

def add_to_6hole(n, nholes, hrad, rng, co, bw, xy0, geometry='cent'):
    if geometry == 'cent':
        timeout2 = 0
        while 1:
            tn = x_choose_y(24-n, nholes-n)
            if timeout2 > tn:
                break
            coord_options2 = co * 1.0
            my_design = design(nholes, hrad)
            for i in range(n):
                my_design.add_hole(xy0, i)
            for i in range(nholes-n):
                hole, coord_options2 = pick_hole_cent(rng, coord_options2)
                my_design.add_hole(hole, n+i)
            my_design.get_xy_m()
            my_design.get_uvs()
            rcheck2 = check_redundancy(my_design, bw)
            if rcheck2 == 0:
                return my_design
            timeout2 += 1
    #if geometry == 'rand':

########################################################################################################
########################################################################################################
########################################################################################################

def make_design(nholes, hrad, ap, bw=0, geometry='cent'):
    rng = np.random.default_rng(seed=None) # set random number generator
    n = 7
    if nholes > n:
        final_design = start_from6(ap.hcmb_coords, n, nholes, hrad, rng, bw, geometry) # makes a 6 hole then adds on
        return final_design
    else:
        final_design = start_from0(nholes, hrad, rng, bw, ap.hcmb_coords, geometry)
        return final_design

########################################################################################################
########################################################################################################
########################################################################################################

def make_design2(nholes, hrad, ap, return_vcoords=False, bw=0, geometry='cent'): # older version
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

def save_design(mask_design):
    n = mask_design.nholes
    np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/' + str(n) + 'hole_xycoords.npy', mask_design.xy_coords_m)
    np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/' + str(n) + 'hole_uvcoords.npy', mask_design.uv_coords)

########################################################################################################
########################################################################################################
########################################################################################################

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
    
