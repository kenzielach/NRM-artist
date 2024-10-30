# piece of code to write out and save the exclusion matrix for Keck
import numpy as np

name = 'Keck_ex_mat.npy'
filepath = '/Users/kenzie/Desktop/NRM-artist/NRM-artist/nrm-artist/' + name

matrix = np.zeros([12, 12])

for i in range(12):
    matrix[i, i-1] = 1
    matrix[i, i] = 1

np.save(filepath, matrix)
print(matrix)