# Test kronmult function against the Matlab version kronmult.m
# T:\PUBLIC\KJ2\QA4EO\Regridding

import numpy as np
from scipy.linalg import toeplitz
from matheo.linear_algebra.tensor_product import TensorProduct

# matrices to kron
Q = {}
Q[0] = np.array(toeplitz([3, 1, 5]))
Q[1] = np.array(toeplitz([4, 5, 2, 8, 1]))


# outside matrix
X = np.linspace(1, 15, 15)


TPclass = TensorProduct()
print(TPclass.kronmult(X, Q))

## If Q is vectors
Q2 = {}
Q2[0] = np.array([3, 1, 5])
Q2[1] = np.array([4, 5, 2, 8, 1])


#TPclass = TensorProduct(Q2, X)
#print(TPclass.kronmult())