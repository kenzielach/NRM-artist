from masks_functs import *

print('Hello! Please enter the number of holes for your mask:')

nholes = int(input())

print('Please enter the hole radius in meters:')

# add exceptions for hole too small, hole too big, too few holes, nholes not an int

hrad = float(input())

print('Thanks! Generating mask design...')

mask_design = make_design(nholes, hrad)

print('Done! Enjoy your mask :)')