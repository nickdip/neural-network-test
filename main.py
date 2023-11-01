import numpy as np
import random


# arrays = [ np.random.rand(1,4)[0] for i in range(10) ]
import numpy as np
import random


# arrays = [ np.random.rand(1,4)[0] for i in range(10) ]

arrays = [[0.74767894, 0.33266191, 0.90619521, 0.00650555],
 [0.7933511,  0.09540726, 0.93216126, 0.26082209],
 [0.22820346, 0.47445412, 0.25426897, 0.40395878],
 [0.84715109, 0.39174264, 0.88434984, 0.61120662],
 [0.77392417, 0.7468356, 0.55454297, 0.57976252],
 [0.66254309, 0.66344524, 0.66609694, 0.03369412],
 [0.7232905,  0.32448762, 0.44126622, 0.97207849],
 [0.97014814, 0.34441813, 0.65367986, 0.95187863],
 [0.81349047, 0.39533349, 0.47569094, 0.60644582],
 [0.04946012, 0.76034437, 0.072437,  0.56172153]]

matrix = []
#fixing index as 3
for i in range(len(arrays)):
    matrix.append(np.array(arrays[i]))

print(matrix)


test_vector = [0.9, 0.3, 0.7, 0.95]

dot_itself = np.dot(test_vector, test_vector)

print(dot_itself)

differences = []

for i in range(len(arrays)):
    differences.append(abs(dot_itself - np.dot(test_vector, matrix[i])))

print(differences.index(min(differences)))
