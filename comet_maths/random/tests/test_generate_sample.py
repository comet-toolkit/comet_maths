"""Tests for classname module"""

"""___Built-In Modules___"""
# import here
from comet_maths.random.generate_sample import *
import numpy as np
import numpy.testing as npt

"""___Third-Party Modules___"""
import unittest

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

x0 = 5
u_x0 = 1

x1 = np.arange(0, 10, 0.5)
u_x1 = np.ones_like(x1)

x2 = 2 * np.ones((5, 6)) + np.random.random((5, 6))
u_x2 = np.ones_like(x2)

x5 = np.random.random((2, 3, 4, 5, 6))
u_x5 = np.ones_like(x5)
corr_5_syst = np.ones((2 * 3 * 4 * 5 * 6, 2 * 3 * 4 * 5 * 6))
corr_5_rand = np.eye(2 * 3 * 4 * 5 * 6)


class TestGenerateSample(unittest.TestCase):
    def test_generate_sample(self):
        sample = generate_sample(1000, [x1, x2], [u_x1, u_x2], ["rand", "syst"], i=0)
        npt.assert_equal(sample.shape, (1000, 20))
        npt.assert_allclose(u_x1, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(np.eye(20), np.corrcoef(sample, rowvar=False), atol=0.15)

        sample = generate_sample(1000, [x1, x2], [u_x1, u_x2], ["rand", "syst"], i=1)
        npt.assert_equal(sample.shape, (1000, 5, 6))
        npt.assert_allclose(u_x2, np.std(sample, axis=0), rtol=0.1)

    def test_generate_error_sample(self):
        sample = generate_error_sample(
            1000, [x1, x2], [u_x1, u_x2], ["rand", "syst"], i=0
        )
        npt.assert_equal(sample.shape, (1000, 20))
        npt.assert_allclose(u_x1, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(np.zeros_like(u_x1), np.mean(sample, axis=0), atol=0.1)
        npt.assert_allclose(np.eye(20), np.corrcoef(sample, rowvar=False), atol=0.15)

        sample = generate_error_sample(
            1000, [x1, x2], [u_x1, u_x2], ["rand", "syst"], i=1
        )
        npt.assert_equal(sample.shape, (1000, 5, 6))
        npt.assert_allclose(u_x2, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(np.zeros_like(u_x2), np.mean(sample, axis=0), atol=0.1)

        sample = generate_error_sample(10000, x5, u_x5, "rand")
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(np.zeros_like(u_x5), np.mean(sample, axis=0), atol=0.1)
        npt.assert_allclose(
            corr_5_rand,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        sample = generate_error_sample(10000, x5, u_x5, corr_5_syst)
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(
            np.zeros_like(x5), np.mean(sample, axis=0), rtol=0.1, atol=0.05
        )
        npt.assert_allclose(
            corr_5_syst,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        sample = generate_error_sample(
            10000, x5, u_x5, {"3": np.eye(5), "0": np.eye(2)}
        )
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(
            np.zeros_like(x5), np.mean(sample, axis=0), rtol=0.1, atol=0.05
        )
        npt.assert_allclose(
            corr_5_rand,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

    def test_generate_sample_same(self):
        sample = generate_sample_same(100, x0)
        npt.assert_equal(sample.shape, (100,))
        npt.assert_equal(sample[:-1], sample[1:])

        sample = generate_sample_same(100, x1)
        npt.assert_equal(sample.shape, (100, 20))
        npt.assert_equal(sample[:-1], sample[1:])

        sample = generate_sample_same(100, x2)
        npt.assert_equal(sample.shape, (100, 5, 6))
        npt.assert_equal(sample[:-1], sample[1:])

    def test_generate_sample_systematic(self):
        sample = generate_sample_systematic(1000, x0, u_x0)
        npt.assert_equal(sample.shape, (1000,))
        npt.assert_allclose(u_x0, np.std(sample, axis=0), rtol=0.1)

        sample = generate_sample_systematic(1000, x1, u_x1)
        npt.assert_equal(sample.shape, (1000, 20))
        npt.assert_allclose(u_x1, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(
            np.ones((20, 20)), np.corrcoef(sample, rowvar=False), atol=0.1
        )

        sample = generate_sample_systematic(1000, x2, u_x2)
        npt.assert_equal(sample.shape, (1000, 5, 6))
        npt.assert_allclose(u_x2, np.std(sample, axis=0), rtol=0.1)

        sample = generate_sample_systematic(
            1000, x2, u_x2, pdf_shape="truncated_gaussian", pdf_params={"min": 0}
        )
        npt.assert_equal(sample.shape, (1000, 5, 6))
        npt.assert_array_less(0, sample)
        npt.assert_allclose(u_x2, np.std(sample, axis=0), rtol=0.1)

        sample = generate_sample_systematic(1000, x2, u_x2, pdf_shape="tophat")
        npt.assert_equal(sample.shape, (1000, 5, 6))
        npt.assert_allclose(u_x2 / 3**0.5, np.std(sample, axis=0), rtol=0.1)

        sample = generate_sample_systematic(1000, x5, u_x5)
        npt.assert_equal(sample.shape, (1000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(
            corr_5_syst, np.corrcoef(sample.reshape((1000, -1)), rowvar=False), atol=0.1
        )

    def test_generate_sample_random(self):
        sample = generate_sample_random(1000, x0, u_x0)
        npt.assert_equal(sample.shape, (1000,))
        npt.assert_allclose(u_x0, np.std(sample, axis=0), rtol=0.1)

        sample = generate_sample_random(
            1000, x0, u_x0, pdf_shape="truncated_gaussian", pdf_params={"max": 4}
        )
        npt.assert_equal(sample.shape, (1000,))
        npt.assert_array_less(sample, 4)

        sample = generate_sample_random(1000, x1, u_x1)
        npt.assert_equal(sample.shape, (1000, 20))
        npt.assert_allclose(u_x1, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(np.eye(20), np.corrcoef(sample, rowvar=False), atol=0.15)

        sample = generate_sample_random(1000, x2, u_x2)
        npt.assert_equal(sample.shape, (1000, 5, 6))
        npt.assert_allclose(u_x2, np.std(sample, axis=0), rtol=0.1)

        sample = generate_sample_random(
            1000, x2, u_x2, pdf_shape="truncated_gaussian", pdf_params={"min": 0}
        )
        npt.assert_equal(sample.shape, (1000, 5, 6))
        npt.assert_array_less(0, sample)

        sample = generate_sample_random(
            1000, x2, u_x2, pdf_shape="truncated_gaussian", pdf_params={"max": 4}
        )
        npt.assert_equal(sample.shape, (1000, 5, 6))
        npt.assert_array_less(sample, 4)

        sample = generate_sample_random(1000, x2, u_x2, pdf_shape="tophat")
        npt.assert_equal(sample.shape, (1000, 5, 6))
        npt.assert_allclose(u_x2 / 3**0.5, np.std(sample, axis=0), rtol=0.1)

        sample = generate_sample_random(10000, x5, u_x5)
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(
            corr_5_rand,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

    def test_generate_sample_cov(self):
        cov_5_rand = cm.convert_corr_to_cov(corr_5_rand, u_x5)
        cov_5_syst = cm.convert_corr_to_cov(corr_5_syst, u_x5)

        sample = generate_sample_cov(10000, x5, cov_5_rand)
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_rand,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        sample = generate_sample_cov(10000, x5, cov_5_syst)
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_syst,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

    def test_generate_sample_correlated(self):
        sample = generate_sample_correlated(10000, x5, u_x5, corr_5_rand)
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_rand,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        sample = generate_sample_correlated(10000, x5, u_x5, corr_5_syst)
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_syst,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

    def test_generate_sample_correlated_dict(self):
        sample = generate_sample_correlated(10000, x5, u_x5, {"3": np.eye(5)})
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_rand,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        sample = generate_sample_correlated(10000, x5, u_x5, {"0": np.eye(2)})
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_rand,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        sample = generate_sample_correlated(
            10000, x5, u_x5, {"3": np.eye(5), "0": np.eye(2)}
        )
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_rand,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        sample = generate_sample_correlated(10000, x5, u_x5, {"0.2": np.eye(8)})
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_rand,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        sample = generate_sample_correlated(
            10000, x5, u_x5, {"0.1.2.3.4": np.ones((720, 720))}
        )
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_syst,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        # sample=generate_sample_correlated(10000,x5,u_x5,{"3.1.2.0.4": np.ones((720,720))})
        # npt.assert_equal(sample.shape,(10000,2,3,4,5,6))
        # npt.assert_allclose(u_x5,np.std(sample,axis=0),rtol=0.1)
        # npt.assert_allclose(x5,np.mean(sample,axis=0),rtol=0.1, atol=0.05)
        # npt.assert_allclose(corr_5_syst,np.corrcoef(sample.reshape((10000,-1)),rowvar=False),atol=0.1)

        sample = generate_sample_correlated(
            10000,
            x5,
            u_x5,
            {
                "3": np.ones((5, 5)),
                "4": np.ones((6, 6)),
                "0": np.ones((2, 2)),
                "1": np.ones((3, 3)),
                "2": np.ones((4, 4)),
            },
        )
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_syst,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        sample = generate_sample_correlated(
            10000,
            x5,
            u_x5,
            {
                "3": "syst",
                "4": "syst",
                "0": "syst",
                "1": "syst",
                "2": "syst",
            },
        )
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_syst,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        sample = generate_sample_correlated(
            10000,
            x5,
            [u_x5, u_x5],
            [
                {
                    "3": "syst",
                    "4": "syst",
                    "0": "syst",
                    "1": "syst",
                    "2": "syst",
                },
                {
                    "3": "syst",
                    "4": "syst",
                    "0": "syst",
                    "1": "syst",
                    "2": "syst",
                },
            ],
            comp_list=True,
        )
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(2**0.5 * u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_syst,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

    def test_correlate_sample_corr(self):
        sample = generate_sample_correlated(10000, x5, u_x5, corr_5_rand)
        sample2 = generate_sample_correlated(10000, x5, u_x5, corr_5_rand)
        corr_betw = np.ones((2, 2))

        sample_corr = correlate_sample_corr([sample, sample2], corr_betw)

        npt.assert_equal(sample_corr[0].shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_equal(sample_corr[1].shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample_corr[0], axis=0), rtol=0.1)
        npt.assert_allclose(
            np.zeros_like(u_x5),
            np.std(sample_corr[0] - sample_corr[1], axis=0),
            atol=0.1,
        )
        npt.assert_allclose(x5, np.mean(sample_corr[0], axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_rand,
            np.corrcoef(sample_corr[0].reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )


if __name__ == "__main__":
    unittest.main()
