"""
Tests for mc propagation class
"""

import unittest

import numpy as np
import numpy.testing as npt

import punpy.utilities.utilities as util

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "14/4/2020"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

class TestMatrixCalculation(unittest.TestCase):
    """
    Class for unit tests
    """

    def test_nearestPD_cholesky(self):
        A = np.ones((10, 10)) + np.diag(np.ones(10))
        B = util.nearestPD_cholesky(A, diff=0.001, corr=False)
        B = util.nearestPD_cholesky(A, diff=0.001, corr=False, return_cholesky=False)
        npt.assert_allclose(A, B, atol=0.06)

        A = np.ones((10, 10))
        B = util.nearestPD_cholesky(A, diff=0.001, corr=True)
        B = util.nearestPD_cholesky(A, diff=0.001, corr=False, return_cholesky=False)
        npt.assert_allclose(A, B, atol=0.06)

        A = np.ones((10, 10)) - np.diag(np.ones(10))
        try:
            B = util.nearestPD_cholesky(A, diff=0.001, corr=False)
        except:
            print("done")


if __name__ == "__main__":
    unittest.main()
