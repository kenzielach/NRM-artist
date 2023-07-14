import numpy as np
import math

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
        self.xy_coords = np.empty([nholes, 2])
        self.uv_coords = np.empty([x_choose_y(self.nholes, 2), 2])
        
    def make_uv_coords(self, uv_coords):
        count = 0
        for i in range(self.nholes):
            xy1 = self.xy_coords[i]
            for j in range(self.nholes):
                if (i == j) or (j < i):
                    continue
                xy2 = self.xy_coords[j]
                u = np.abs(xy1[0] - xy2[0])
                v = np.abs(xy1[1] - xy2[1])
                uv_coords[count] = [u, v]
                count += 1
        self.uv_coords = uv_coords
        return self.uv_coords

    def get_uvs(self):
        self.uv_coords = self.make_uv_coords(self.uv_coords)
        return self.uv_coords
                    