"""Tests for classname module"""

"""___Built-In Modules___"""
# import here
from comet_maths.generate_sample.generate_sample import *
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
np.random.seed(12345)


class TestGenerateSample(unittest.TestCase):
    def test_generate_sample(self):
        sample = generate_sample(1000, [x1, x2], [u_x1, u_x2], ["rand", "syst"], i=0)
        npt.assert_equal(sample.shape, (1000, 20))
        npt.assert_allclose(u_x1, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(np.eye(20), np.corrcoef(sample, rowvar=False), atol=0.15)

        sample = generate_sample(1000, [x1, x2], [u_x1, u_x2], ["rand", "syst"], i=1)
        npt.assert_equal(sample.shape, (1000, 5, 6))
        npt.assert_allclose(u_x2, np.std(sample, axis=0), rtol=0.1)

        self.assertRaises(
            ValueError,
            generate_error_sample,
            1000,
            [x1, x2],
            [-u_x1, -u_x2],
            ["rand", "syst"],
            i=1,
        )

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
        assert sample.dtype == int

        sample = generate_sample_same(100, x1)
        npt.assert_equal(sample.shape, (100, 20))
        npt.assert_equal(sample[:-1], sample[1:])

        sample = generate_sample_same(100, x2)
        npt.assert_equal(sample.shape, (100, 5, 6))
        npt.assert_equal(sample[:-1], sample[1:])

        sample = generate_sample_same(100, "test")
        npt.assert_equal(sample.shape, (100, ))
        npt.assert_equal(sample[:-1], sample[1:])
        assert sample.dtype.type is np.str_

    def test_generate_sample_systematic(self):
        np.random.seed(123456)
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

    def test_generate_sample_corr(self):
        cov_5_rand = cm.convert_corr_to_cov(corr_5_rand, u_x5)
        cov_5_syst = cm.convert_corr_to_cov(corr_5_syst, u_x5)

        sample = generate_sample_corr(10000, x5, u_x5, corr_5_rand)
        npt.assert_equal(sample.shape, (10000, 2, 3, 4, 5, 6))
        npt.assert_allclose(u_x5, np.std(sample, axis=0), rtol=0.1)
        npt.assert_allclose(x5, np.mean(sample, axis=0), rtol=0.1, atol=0.05)
        npt.assert_allclose(
            corr_5_rand,
            np.corrcoef(sample.reshape((10000, -1)), rowvar=False),
            atol=0.1,
        )

        sample = generate_sample_corr(10000, x5, u_x5, corr_5_syst)
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

        errcorrx = np.eye(3)
        errcorrx[0, 2] = 0.5
        errcorrx[2, 0] = 0.5
        a = cm.generate_sample(10000, np.zeros(3), np.ones(3), errcorrx)
        b = cm.generate_sample(
            10000,
            np.zeros(3),
            np.ones(3),
            0.5 * np.ones_like(errcorrx) + 0.5 * np.eye(len(errcorrx)),
        )
        c = cm.correlate_sample_corr(
            [a, b],
            np.array([[1, 0.4], [0.4, 1]]),
            np.zeros((2, 3)),
            np.ones((2, 3)),
            iterate_sample=True,
        )
        corr_c = cm.calculate_corr(np.concatenate(c, axis=1))
        npt.assert_allclose(
            corr_c,
            np.array(
                [
                    [1, 0, 0.5, 0.4, 0.1, 0.4 * 0.5],
                    [0, 1, 0, 0.1, 0.4, 0.1],
                    [0.5, 0, 1, 0.4 * 0.5, 0.1, 0.4],
                    [0.4, 0.25 * 0.4, 0.4 * 0.5, 1.0, 0.5, 0.5],
                    [0.1, 0.4, 0.1, 0.5, 1.0, 0.5],
                    [0.4 * 0.5, 0.1, 0.4, 0.5, 0.5, 1.0],
                ]
            ),
            atol=0.05,
        )

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

    def test_generate_few_samples(self):
        x_HR = np.arange(-0.5, 4.0, 0.09)
        y_HR = np.zeros_like(x_HR)
        u_y_HR_syst = 0.9 * np.ones_like(y_HR)
        u_y_HR_rand = np.abs(0.02)
        cov_y_HR = cm.convert_corr_to_cov(
            np.ones((len(y_HR), len(y_HR))), u_y_HR_syst
        ) + cm.convert_corr_to_cov(np.eye(len(y_HR)), u_y_HR_rand)
        corr_y_HR = cm.correlation_from_covariance(cov_y_HR)
        u_y_HR = cm.uncertainty_from_covariance(cov_y_HR)

        y_HR2 = cm.generate_sample(1, y_HR, u_y_HR, corr_x=corr_y_HR)

        assert np.std(y_HR2 - y_HR) < 0.05

        y_HR2 = cm.generate_sample(3, y_HR, u_y_HR, corr_x=corr_y_HR)

        npt.assert_allclose(np.std(y_HR2 - y_HR, axis=1), 0.02, atol=0.02)


if __name__ == "__main__":
    unittest.main()
