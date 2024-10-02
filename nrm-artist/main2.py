from masks_functs import *
from design_class import *
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import math

# 6 hole manx
#nholes = 6
#hrad = 0.5
aperture = pyfits.getdata('/Users/kenzie/Desktop/CodeAstro/planet-guts/keck_aperture.fits') # set Keck primary aperture
#manx_array = np.array([[1013, 545], [155, 410], [301, 140], [857, 275], [301, 950], [623, 950]])
#manx_design = design(nholes, hrad)
#manx_design.xy_coords_cm = manx_array
#manx_design.get_xy_m()
#manx_design.get_uvs()

#plot_design(manx_design, aperture)

#np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/6hole_manx_xycoords.npy', manx_design.xy_coords_m)
#np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/6hole_manx_uvcoords.npy', manx_design.uv_coords)

# 7 hole asymmetric
#hrad = 0.65
#nholes = 7
#asym_array = np.array([[77, 545], [467, 140], [935, 410], [301, 950], [789, 950], [857, 815], [701, 815]])
#asym_design = design(nholes, hrad)
#asym_design.xy_coords_cm = asym_array
#asym_design.get_xy_m()
#asym_design.get_uvs()

#plot_design(asym_design, aperture)

#np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/7hole_asym_xycoords.npy', asym_design.xy_coords_m)
#np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/7hole_asym_uvcoords.npy', asym_design.uv_coords)

#thing = design(9, 0.5)
#thing.xy_coords_m = np.load('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/nice_9hole_xycoords.npy')
#thing.uv_coords = np.load('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/nice_9hole_uvcoords.npy')
#plot_design(thing, aperture)
#print(thing.xy_coords_m)

# 9 hole NIRC2
#nholes = 9
#hrad = 0.7
#thing = design(nholes, hrad)
#nirc2 = np.array([[857, 545], [77, 545], [155, 410], [389, 815], [389, 275], [857, 275], [623, 950], [789, 950], [789, 140]])
#nirc22 = nirc2 - [545, 545]
#np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/SPIE_2024/9hole_nirc2_xycoords_cent.npy', nirc22)
#thing.xy_coords_cm = nirc2
#thing.get_xy_m()
#thing.get_uvs()
#plot_design(thing, aperture)
#np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/9hole_nirc2_xycoords.npy', thing.xy_coords_m)
#np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/9hole_nirc2_uvcoords.npy', thing.uv_coords)
#print(check_redundancy(thing))

# other 9 hole
nholes = 9
hrad = 0.45
thing = design(nholes, hrad)
#array = np.array([[], [], [], [], [], [], []])
#thing2 = design(nholes, hrad)
#array1 = np.array([[155, 410], [155, 680], [233, 815], [467, 950], [935, 680], [1013, 545], [789, 140], [467, 140], [301, 140]])
#array2 = np.array([[155, 410], [155, 680], [233, 275], [467, 950], [301, 950], [1013, 545], [935, 410], [467, 140], [789, 950]])
thing.xy_coords_m = np.loadtxt('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/SPIE_2024/final_mask_files/unrotated/m/9hole1_hp_m.txt') #+ [5.45, 5.45]
#thing.xy_coords_cm = array
#thing2.xy_coords_cm = array2
#thing.xy_coords_m[2, 0] = 3.1
#thing.xy_coords_m[4, 0] = 3.1
thing.xy_coords_cm = thing.xy_coords_m * 100
thing.get_uvs()
#thing2.xy_coords_m = thing2.xy_coords_cm / 100
#thing2.get_uvs()
#print(check_redundancy(thing))
#print(thing.xy_coords_m)
plot_design(thing, aperture)
#plot_ps(thing)
#np.savetxt('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/SPIE_2024/6hole_manx_xycoords_cent.txt', thing.xy_coords_m - [5.45, 5.45])
#np.save('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/SPIE_2024/9hole_nirc2_xycoords_cent.npy', thing.xy_coords_m - [5.45, 5.45])