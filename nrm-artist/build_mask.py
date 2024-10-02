from masks_functs import *
from design_class import *
from aperture_class import *
from init_checks import *
import matplotlib.pyplot as plt

def build_mask(nholes, hrad, name):
    init_checks(nholes, hrad, name) # make sure inputs are ok
    ap = aperture(tn=name) # initialize aperture
    design = make_design(nholes, hrad, ap, geometry='cent') # make design
    return design, ap