import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array = np.reshape(matrix, newshape=np.size(matrix))
print(array.shape)
print(array)

matrix_back = np.reshape(array, newshape=(3, 3))
print(matrix_back)
