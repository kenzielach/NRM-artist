from masks_functs import *

print('Hello! Please enter the number of holes for your mask:')
nholes = int(input())
check_valid_nholes(nholes)

print('Please enter the hole radius in meters:')
hrad = float(input())
check_valid_hrad(hrad)

print('Thanks! Generating mask design...')

if nholes > 6:
    mask_design, vcoords = make_design(6, hrad, return_vcoords=True)
    diff = nholes - 6
    mask_design = add_to_design(mask_design, diff, vcoords)    
    save_design(mask_design)
    print('Done! Enjoy your mask :)')

else:
    mask_design = make_design(nholes, hrad)
    #print('Would you like to save your mask as an .npy (y/n)?')
    #ans = str(input())
    #if ans == 'y':
    print("Yay! Mask design is non-redundant. Plotting design...")
    #plot_design(mask_design, aperture)
    save_design(mask_design)
    print('Done! Enjoy your mask :)')