from masks_functs import *
from design_class import *
from aperture_class import *
from init_checks import *
import matplotlib.pyplot as plt

def build_mask(nholes, hrad, tn):
    init_checks(nholes, hrad)
    ap = aperture(tn)
    design = make_design(nholes, hrad, ap)
    plot_design(design, ap)