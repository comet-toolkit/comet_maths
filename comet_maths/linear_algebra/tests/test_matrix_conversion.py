"""
Tests for mc propagation class
"""

import unittest
import copy
import numpy as np
import numpy.testing as npt
import comet_maths as cm

from comet_maths.linear_algebra.matrix_conversion import *

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "14/4/2020"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

dim_sizes = {"x": 3, "y": 2, "z": 4, "time": 4}

errcorrx = np.eye(dim_sizes["x"])
errcorrx[0, 2] = 0.5
errcorrx[2, 0] = 0.5

u_x = 10 * np.ones(3)
cov_x = np.array([[100, 0, 50], [0, 100, 0], [50, 0, 100]])

errcorry = np.ones((dim_sizes["y"], dim_sizes["y"]))
errcorry[0, 1] = 0.3
errcorry[1, 0] = 0.3

errcorrz = np.ones((dim_sizes["z"], dim_sizes["z"]))

errcorrt = np.eye(dim_sizes["time"])

errcorrxy = np.array(
    [
        [1.0, 0.3, 0.0, 0.0, 0.5, 0.15],
        [0.3, 1.0, 0.0, 0.0, 0.15, 0.5],
        [0.0, 0.0, 1.0, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.3, 1.0, 0.0, 0.0],
        [0.5, 0.15, 0.0, 0.0, 1.0, 0.3],
        [0.15, 0.5, 0.0, 0.0, 0.3, 1.0],
    ]
)

errcorryx = np.array(
    [
        [1.0, 0.0, 0.5, 0.3, 0.0, 0.15],
        [0.0, 1.0, 0.0, 0.0, 0.3, 0.0],
        [0.5, 0.0, 1.0, 0.15, 0.0, 0.3],
        [0.3, 0.0, 0.15, 1.0, 0.0, 0.5],
        [0.0, 0.3, 0.0, 0.0, 1.0, 0.0],
        [0.15, 0.0, 0.3, 0.5, 0.0, 1.0],
    ]
)


class TestMatrixConversion(unittest.TestCase):
    """
    Class for unit tests
    """

    def test_correlation_from_covariance(self):
        npt.assert_equal(correlation_from_covariance(cov_x), errcorrx)

    def test_uncertainty_from_covariance(self):
        npt.assert_equal(uncertainty_from_covariance(cov_x), u_x)

    def test_convert_corr_to_cov(self):
        npt.assert_equal(convert_corr_to_cov(errcorrx, u_x), cov_x)

    def test_convert_cov_to_corr(self):
        npt.assert_equal(convert_cov_to_corr(cov_x, u_x), errcorrx)

    def test_calculate_flattened_corr(self):
        flat_corr = calculate_flattened_corr(
            [errcorrx, 0.5*np.ones_like(errcorrx)+0.5 * np.eye(len(errcorrx))], np.array([[1, 0.4], [0.4, 1]])
        )

        npt.assert_equal(
            flat_corr,
            np.array(
                [
                    [1      , 0     , 0.5    , 0.4       , 0.1   , 0.4*0.5],
                    [0      , 1     , 0      , 0.1       , 0.4   , 0.1  ],
                    [0.5    , 0     , 1      , 0.4*0.5   , 0.1   , 0.4  ],
                    [0.4    ,0.25*0.4,0.4*0.5, 1.0       , 0.5   , 0.5],
                    [0.1    , 0.4   , 0.1    , 0.5       , 1.0   , 0.5],
                    [0.4*0.5, 0.1   , 0.4    , 0.5       , 0.5   , 1.0],
                ]
            ),
        )

    def test_separate_flattened_corr(self):
        flat_cor = np.array(
                [
                    [1      , 0     , 0.5    , 0.4       , 0.1   , 0.4*0.5],
                    [0      , 1     , 0      , 0.1       , 0.4   , 0.1  ],
                    [0.5    , 0     , 1      , 0.4*0.5   , 0.1   , 0.4  ],
                    [0.4    ,0.25*0.4,0.4*0.5, 1.0       , 0.5   , 0.5],
                    [0.1    , 0.4   , 0.1    , 0.5       , 1.0   , 0.5],
                    [0.4*0.5, 0.1   , 0.4    , 0.5       , 0.5   , 1.0],
                ]
            )
        corrs, corr_betw = separate_flattened_corr(flat_cor, 2)
        npt.assert_equal(corrs[0], errcorrx)
        npt.assert_equal(corrs[1], 0.5*np.ones_like(errcorrx)+0.5 * np.eye(len(errcorrx)))
        npt.assert_allclose(corr_betw, np.array([[1, 0.4], [0.4, 1]]), atol=0.01)

    def test_change_order_errcorr_dims(self):
        errcorryx2 = change_order_errcorr_dims(
            errcorrxy, ["x", "y"], ["y", "x"], dim_sizes
        )
        npt.assert_equal(errcorryx, errcorryx2)

    def test_expand_errcorr_dims_2dims(self):
        outcorr_x = expand_errcorr_dims(errcorrx, "x", ["y", "x"], dim_sizes)
        outcorr_y = expand_errcorr_dims(errcorry, "y", ["y", "x"], dim_sizes)

        errcorryx2 = np.dot(outcorr_y, outcorr_x)

        npt.assert_equal(errcorryx, errcorryx2)

        outcorr_x = expand_errcorr_dims(errcorrx, "x", ["x", "y"], dim_sizes)
        outcorr_y = expand_errcorr_dims(errcorry, "y", ["x", "y"], dim_sizes)

        errcorrxy2 = np.dot(outcorr_x, outcorr_y)

        npt.assert_equal(errcorrxy, errcorrxy2)

    def test_expand_errcorr_dims_3dims(self):
        outcorr2_x = expand_errcorr_dims(errcorrx, "x", ["z", "x", "y"], dim_sizes)
        outcorr2_y = expand_errcorr_dims(errcorry, "y", ["z", "x", "y"], dim_sizes)
        outcorr2_z = expand_errcorr_dims(errcorrz, "z", ["z", "x", "y"], dim_sizes)

        corrdims = ["x", "y"]
        outcorr2_xy = expand_errcorr_dims(
            errcorrxy, corrdims, ["z", "x", "y"], dim_sizes
        )

        corrdims = ["y", "x"]
        outcorr2_yx = expand_errcorr_dims(
            errcorryx, corrdims, ["z", "x", "y"], dim_sizes
        )

        npt.assert_equal(
            np.dot(outcorr2_xy, outcorr2_z),
            np.dot(np.dot(outcorr2_x, outcorr2_y), outcorr2_z),
        )
        npt.assert_equal(
            np.dot(outcorr2_yx, outcorr2_z),
            np.dot(np.dot(outcorr2_x, outcorr2_y), outcorr2_z),
        )

    def test_expand_errcorr_dims_4dims(self):
        outcorr2_x = expand_errcorr_dims(
            errcorrx, "x", ["x", "y", "z", "time"], dim_sizes
        )
        outcorr2_y = expand_errcorr_dims(
            errcorry, "y", ["x", "y", "z", "time"], dim_sizes
        )
        outcorr2_z = expand_errcorr_dims(
            errcorrz, "z", ["x", "y", "z", "time"], dim_sizes
        )
        outcorr2_t = expand_errcorr_dims(
            errcorrt, "time", ["x", "y", "z", "time"], dim_sizes
        )

        corrdims = ["x", "y"]
        outcorr2_xy = expand_errcorr_dims(
            errcorrxy, corrdims, ["x", "y", "z", "time"], dim_sizes
        )

        corrdims = ["y", "x"]
        outcorr2_yx = expand_errcorr_dims(
            errcorryx, corrdims, ["x", "y", "z", "time"], dim_sizes
        )

        npt.assert_equal(
            np.dot(np.dot(outcorr2_xy, outcorr2_z), outcorr2_t),
            np.dot(np.dot(np.dot(outcorr2_x, outcorr2_y), outcorr2_z), outcorr2_t),
        )
        npt.assert_equal(
            np.dot(np.dot(outcorr2_yx, outcorr2_z), outcorr2_t),
            np.dot(np.dot(np.dot(outcorr2_x, outcorr2_y), outcorr2_z), outcorr2_t),
        )


if __name__ == "__main__":
    unittest.main()
