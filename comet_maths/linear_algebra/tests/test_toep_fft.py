# Test FFT Toeplitz mult against PMH Matlab version KTMfun.m
# T:\PUBLIC\KJ2\QA4EO\Regridding

import numpy as np
import unittest
from matheo.linear_algebra.Toeplitz import Toeplitz

K = np.array([9, 1, 3, 2])
x = np.array([5, 1, 4, 3])
x2 = np.array([[5, 1, 4, 3], [3, 5, 2, 1]])


class TestToeplitz(unittest.TestCase):

    def test_toepfftmult(self):
        Tclass = Toeplitz()

        expected_result = np.array([[64., 27., 55., 44.],
                                    [40., 53., 33., 32.]])
        result = Tclass.toepfftmult(x2, K)

        np.testing.assert_array_almost_equal(result, expected_result, decimal=3)


if __name__ == "__main__":
    unittest.main()
