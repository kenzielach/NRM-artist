from masks_functs import *
from design_class import *
from aperture_class import *
from init_checks import *
import matplotlib.pyplot as plt

def build_mask(nholes, hrad, tname, geometry='cent', method='guess'):
    if method == 'guess':
        init_checks(nholes, hrad, tname) # make sure inputs are ok
        ap = aperture(tn=tname) # initialize aperture
        design = make_design(nholes, hrad, ap, geometry) # make design
        return design, ap
    if method == 'TC':
        init_checks(nholes, hrad, tname)
        ap = aperture(tn=tname)
        