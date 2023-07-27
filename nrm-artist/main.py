from masks_functs import *

print('Hello! Please enter the number of holes for your mask:')
nholes = int(input())
check_valid_nholes(nholes)

print('Please enter the hole radius in meters:')
hrad = float(input())
check_valid_hrad(hrad)

print('Thanks! Generating mask design...')
mask_design = make_design(nholes, hrad)
print('Would you like to save your mask as an .npy (y/n)?')
ans = str(input())
if ans == 'y':
    save_design(mask_design)
print('Done! Enjoy your mask :)')