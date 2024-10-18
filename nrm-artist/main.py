from masks_functs import *
from build_mask import *

#print('Thanks! Generating mask design...')
print('making mask...')
design, aperture = build_mask(9, 0.5, 'keck', 'TC')
print('plotting...')
plot_design(design, aperture)