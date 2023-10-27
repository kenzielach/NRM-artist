from masks_functs import *

print('Hello! Please enter the number of holes for your mask:')
nholes = int(input())
check_valid_nholes(nholes)

print('Please enter the hole radius in meters:')
hrad = float(input())
check_valid_hrad(hrad)

print('Generating mask design...')

if nholes > 6:
    count = 1
    while 1:
        mask_design, vcoords = make_design(6, hrad, return_vcoords=True)
        diff = nholes - 6
        mask_design = add_to_design(mask_design, diff, vcoords) 
        print("iteration # " + str(count))   
        if mask_design.nholes == nholes:
            break
        count += 1
    print('Would you like to save your mask as an .npy (y/n)?')
    ans = str(input())
    if ans == 'y':
        save_design(mask_design)

else:
    mask_design = make_design(nholes, hrad)
    print('Would you like to save your mask as an .npy (y/n)?')
    ans = str(input())
    if ans == 'y':
        save_design(mask_design)