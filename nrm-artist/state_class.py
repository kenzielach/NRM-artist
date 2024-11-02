import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import scipy

class state:
    def up_level(self):
        self.level += 1
        return self.level

    def down_level(self):
        self.level -= 1
        return self.level

    def __init__(self, nholes, a):
        self.level = 0
        self.mcoords = np.ones(a)
        self.level_arrs = []
        self.level_attempts = np.zeros(nholes)

    def up_level_attempts(self):
        self.level_attempts[self.level] += 1
        return self.level_attempts

    def reset_level(self):
        self.level_arrs[self.level] = 0
        self.level_attempts[self.level] = 0
        return self.level_arrs, self.level_attempts

    def add_to_level_arrs(self, hns):
        self.level_arrs.append([hns])
        return self.level_arrs

    def add(self, ind):
        self.mcoords[ind] = 0
        return self.mcoords
    
    def remove(self, ind):
        self.mcoords[ind] = 1
        return self.mcoords