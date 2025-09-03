import numpy as np
a = np.full((2,3),69)
b = np.full((3,2),4)
print(np.matmul(a,b))
c = np.identity(3)
print(np.linalg.det(c))