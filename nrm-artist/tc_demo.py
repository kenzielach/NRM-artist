import numpy as np
from aperture_class import *
from triple_correlation import *

main_dir = '/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/'
keck = aperture('keck')
coords = np.loadtxt(main_dir + 'masks/SPIE_2024/final_mask_files/unrotated/m/9hole1_hp_m.txt')
mask = make_mask_from_coords(coords)
print('finished making mask, working on tc...')
mask_tc = tc(mask)
mask_tc_subframe = subframe_tc(mask_tc)
#plot_mask(mask_tc_subframe*keck.file)