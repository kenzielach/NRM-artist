import math
import numpy as np
import matplotlib.pyplot as plt

def x_choose_y(n, k):
    """ x choose y

    Performs x choose y calculation.

    Args: 
        n (float): x
        k (float): y
    
    Returns:
        float: Result of calculation.
    """
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

class design:
    def __init__(self, nholes, hrad):
        self.nholes = nholes
        self.hrad = hrad
        self.xy_coords_cm = np.empty([nholes, 2])
        self.xy_coords_m = np.empty([nholes, 2])
        self.uv_coords = np.empty([x_choose_y(self.nholes, 2), 2])
        self.uv_coords_full = np.empty([x_choose_y(self.nholes, 2) * 2, 2])
        
    def make_uv_coords(self, uv_coords, uv_coords_full):
        count = 0
        for i in range(self.nholes):
            xy1 = self.xy_coords_m[i]
            for j in range(self.nholes):
                if (i == j) or (j < i):
                    continue
                xy2 = self.xy_coords_m[j]
                u = xy1[0] - xy2[0]
                v = xy1[1] - xy2[1]
                uv_coords[count] = [u, v]
                count += 1
        self.uv_coords = uv_coords
        self.uv_coords_full =  np.append(uv_coords, -uv_coords, axis=0)
        return self.uv_coords, self.uv_coords_full

    def get_uvs(self):
        self.uv_coords, self.uv_coords_full = self.make_uv_coords(self.uv_coords, self.uv_coords_full)
        return self.uv_coords, self.uv_coords_full
    
    def make_xy_coords_m(self, xy_coords_cm):
        xy_coords_m = xy_coords_cm / 100
        self.xy_coords_m = xy_coords_m
        return self.xy_coords_m
    
    def get_xy_m(self):
        self.xy_coords_m = self.make_xy_coords_m(self.xy_coords_cm)
        return self.xy_coords_m
    
    #def plot_mask(self):

                    
    def plot_uv(self):
        uv_plot = np.zeros([1090, 1090])
        hcoords = self.uv_coords_full * 100
        for i in range(1090):
            for j in range(1090):
                for a in range(len(hcoords)):
                    if np.sqrt(((i - 545) - hcoords[a, 0])**2 + ((j - 545) - hcoords[a, 1])**2) < (100 * self.hrad):
                        uv_plot[i, j] = 0
    
        plt.figure()
        plt.imshow(uv_plot)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def add_hole(self, hole, ind): # must give hole coords in m
       self.xy_coords_cm[ind] = hole
       self.xy_coords_m[ind] = self.xy_coords_cm[ind] / 100