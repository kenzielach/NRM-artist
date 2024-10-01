from masks_functs import *
from build_mask import *

#print('Hello! Please enter the number of holes for your mask:')
nholes = 5 #int(input())

#print('Please enter the hole radius in meters:')
hrad = 0.6 #float(input())

#print('Enter name of telescope:')
tn = 'keck' # str(input())

#print('Thanks! Generating mask design...')
build_mask(nholes, hrad, tn)