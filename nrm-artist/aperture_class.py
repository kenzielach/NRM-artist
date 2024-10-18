import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np

class aperture: 
    def load_the_file(fname):
        return pyfits.getdata(fname)
    
    def load_hcmb_file(fname):
        return np.loadtxt(fname)
    
    def __init__(self, tn='null'):#, t_info=[]):
        #if tn == 'null': # come back to this later
        #    self.ftf = t_info[0]
        #    self.gap = t_info[1]
        #    self.nsegs = t_info[2]
        #    self.file = load_the_file(t_info[3])
        #    self.hcmb_coords = load_hcmb_file(t_info[4])

        if tn == 'keck':
            self.ftf = 1.2 # segment flat-to-flat
            self.gap = 0.004 # gap between segments
            self.nsegs = 36 # number of segments
            self.file = pyfits.getdata('/Users/kenzie/Desktop/CodeAstro_2023/planet-guts/keck_aperture.fits')
            self.hcmb_coords = np.loadtxt('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/old/vcoords_keck9hole.txt') # honeycomb coords
            self.fmatrix = np.load('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/forbidden_matrix.npy')

    def plot_ap(self):
        plt.figure()
        plt.imshow(self.file)
        plt.show()
