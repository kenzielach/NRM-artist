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

    def add_deadend(self, decoord):
        self.deadends.append(decoord)
        return self.deadends

    def __init__(self):
        self.level = 0
        self.mcoords = []
        self.deadends = []