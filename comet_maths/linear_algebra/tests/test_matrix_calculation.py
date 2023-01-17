"""
Tests for mc propagation class
"""

import unittest

import numpy as np
import numpy.testing as npt
import comet_maths as cm

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
        B = cm.nearestPD_cholesky(A, diff=0.001, corr=False)
        B = cm.nearestPD_cholesky(A, diff=0.001, corr=False, return_cholesky=False)
        npt.assert_allclose(A, B, atol=0.06)

        A = np.ones((10, 10))
        B = cm.nearestPD_cholesky(A, diff=0.001, corr=True)
        B = cm.nearestPD_cholesky(A, diff=0.001, corr=False, return_cholesky=False)
        npt.assert_allclose(A, B, atol=0.06)

        A = np.ones((10, 10)) - np.diag(np.ones(10))
        try:
            B = cm.nearestPD_cholesky(A, diff=0.001, corr=False)
        except:
            print("done")

    def test_calculate_corr(self):
        x5 = np.random.random((2, 3, 4, 5, 6))
        u_x5 = np.ones_like(x5)
        corr_5_syst = np.ones((2 * 3 * 4 * 5 * 6, 2 * 3 * 4 * 5 * 6))
        corr_5_rand = np.eye(2 * 3 * 4 * 5 * 6)
        sample = cm.generate_sample(
            10000, x5, u_x5, corr_5_syst
        )  # ((corr_5_syst+corr_5_rand)/2))
        npt.assert_allclose(
            np.ones((2, 2)), cm.calculate_corr(sample, corr_dims=0), atol=0.06
        )
        npt.assert_allclose(
            np.ones((3, 3)), cm.calculate_corr(sample, corr_dims=1), atol=0.06
        )
        npt.assert_allclose(
            np.ones((4, 4)), cm.calculate_corr(sample, corr_dims=2), atol=0.06
        )
        npt.assert_allclose(
            np.ones((5, 5)), cm.calculate_corr(sample, corr_dims=3), atol=0.06
        )
        npt.assert_allclose(
            np.ones((6, 6)), cm.calculate_corr(sample, corr_dims=4), atol=0.06
        )
        npt.assert_allclose(
            np.ones((6, 6)), cm.calculate_corr(sample, corr_dims="4"), atol=0.06
        )
        npt.assert_allclose(
            np.ones((30, 30)), cm.calculate_corr(sample, corr_dims="3.4"), atol=0.06
        )
        corr2 = cm.calculate_corr(sample, corr_dims=[3, 4])
        npt.assert_allclose(np.ones((5, 5)), corr2[0], atol=0.06)
        npt.assert_allclose(np.ones((6, 6)), corr2[1], atol=0.06)


if __name__ == "__main__":
    unittest.main()
