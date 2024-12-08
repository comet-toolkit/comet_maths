"""Tests for interpolation module"""

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

"""___Built-In Modules___"""
import comet_maths as cm

"""___Third-Party Modules___"""
import unittest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import time


"""___NPL Modules___"""
import punpy

__author__ = ["Pieter De Vis <pieter.de.vis@npl.co.uk>"]
__all__ = []


def function1(x):
    return 15 * x - 20


def function2(x):
    """The function to predict."""
    return x * np.sin(x * 10)


class TestInterpolation(unittest.TestCase):
    def test_interpolation_1d(self):
        xi = np.arange(0, 3.0, 0.2)
        yi = function2(xi)
        u_yi = 0.05 * np.abs(yi) + 0.01
        yi = cm.generate_sample(1, yi, u_yi, corr_x="rand")

        x = np.array([0.33333, 0.666666, 1, 1.33333, 1.66666, 2, 2.3333])
        # t1=time.time()
        y = cm.interpolate_1d(xi, yi, x, method="cubic")
        y2 = cm.interpolate_1d(xi, yi, x, method="quadratic")
        # t2=time.time()
        # print("t2",t2-t1)
        y, u_y, corr_y = cm.interpolate_1d(
            xi,
            yi,
            x,
            method="cubic",
            add_model_error=False,
            return_uncertainties=True,
            return_corr=True,
        )

        npt.assert_allclose(y, y2, rtol=2 * np.max(u_y / y))
        # t3=time.time()
        # print("t3",t3-t2)

        xi = np.arange(0, 3.0, 0.2)
        yi_2d = function2(xi)[:, None] * np.ones((len(xi), 10))
        x = np.array([0.33333, 0.666666, 1, 1.33333, 1.66666, 2, 2.3333])
        y_2d = cm.interpolate_1d(xi, yi_2d, x, method="cubic")

        npt.assert_allclose(
            y_2d, y[:, None] * np.ones((len(x), 10)), rtol=2 * np.max(u_y / y)
        )

        xx = np.arange(0, 2.5, 0.01)
        yy, u_yy, corr_yy = cm.interpolate_1d(
            xi,
            yi,
            xx,
            method="gpr",
            u_y_i=u_yi,
            corr_y_i="rand",
            min_scale=0.3,
            add_model_error=False,
            return_uncertainties=True,
            return_corr=True,
        )
        # t4=time.time()
        # print("t4",t4-t3)

        yy2, u_yy2, corr_yy2 = cm.interpolate_1d(
            xi,
            yi,
            xx,
            method="cubic",
            u_y_i=u_yi,
            corr_y_i="rand",
            min_scale=0.3,
            add_model_error=False,
            return_uncertainties=True,
            return_corr=True,
        )
        # t5=time.time()
        # print("t5",t5-t4)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xx, function2(xx), "b", label="true values")
        ax.errorbar(xi, yi, yerr=u_yi, fmt="bo", ls=None, label="observed values")
        ax.plot(
            xx,
            cm.interpolate_1d(xi, yi, xx, method="cubic"),
            "r:",
            label="cubic interpolation",
        )
        ax.fill_between(
            xx,
            yy2 - 1.9600 * u_yy2,
            (yy2 + 1.9600 * u_yy2),
            alpha=0.25,
            fc="r",
            ec="None",
            label="95% confidence interval",
            lw=0,
        )
        ax.plot(xx, yy, "g", label="gpr interpolation")
        ax.fill_between(
            xx,
            yy - 1.9600 * u_yy,
            (yy + 1.9600 * u_yy),
            alpha=0.25,
            fc="g",
            ec="None",
            label="95% confidence interval",
            lw=0,
        )
        ax.legend()
        fig.savefig("interpolation_test_1d.png", bbox_inches="tight")

        fig2 = plt.figure(figsize=(10, 5))
        ax = fig2.add_subplot(1, 2, 1)
        ax2 = fig2.add_subplot(1, 2, 2)
        p1 = ax.imshow(corr_yy, vmin=-1, vmax=1, cmap="bwr")
        ax.set_title("gpr interpolation")
        p2 = ax2.imshow(corr_yy2, vmin=-1, vmax=1, cmap="bwr")
        ax2.set_title("cubic interpolation")
        # cb_ax = fig.add_axes([0.9, 0.2, 0.04, 1.0])
        fig2.colorbar(p2)
        fig2.savefig("interpolation_test_1d_corrs.png", bbox_inches="tight")

    def test_interpolation_1d_along_example(self):
        np.random.seed(1234567)
        xi = np.arange(0, 3.0, 0.2)
        yi = function2(xi)
        u_yi = np.abs(0.01 * yi)
        yi = cm.generate_sample(1, yi, u_yi, corr_x="rand").squeeze()
        x_HR = np.arange(0, 3.0, 0.036)
        y_HR = function2(x_HR)
        u_y_HR = 0.9 * np.ones_like(y_HR)
        corr_y_HR = np.ones((len(y_HR), len(y_HR)))  # + 0.05 * np.eye(len(y_HR))
        y_HR = cm.generate_sample(1, y_HR, u_y_HR, corr_x="syst")

        xx = np.arange(0, 2.5, 0.01)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xx, function2(xx), "b", label="true values")
        ax.plot(xi, yi, "ro", label="LR observed values")
        ax.plot(x_HR, y_HR, "mo", label="HR observed values")
        ax.plot(
            xx,
            cm.interpolate_1d(xi, yi, xx, method="cubic"),
            "r:",
            label="cubic interpolation",
        )
        ax.plot(
            xx,
            cm.interpolate_1d(x_HR, y_HR, xx, method="cubic"),
            "m:",
            label="cubic interpolation HR",
        )
        ax.plot(
            xx,
            cm.interpolate_1d_along_example(
                xi,
                yi,
                x_HR,
                y_HR,
                xx,
                method="cubic",
                method_hr="cubic",
                relative=False,
            ),
            "r-.",
            label="cubic interpolation along example",
        )
        ax.plot(
            xx,
            cm.interpolate_1d_along_example(
                xi, yi, x_HR, y_HR, xx, method="gpr", method_hr="gpr", relative=False
            ),
            "g-.",
            label="gpr interpolation along example",
        )
        ax.legend()
        fig.savefig("interpolation_test_1d_along_example.png", bbox_inches="tight")

    def test_interpolation_1d_along_example_unc(self):
        np.random.seed(123456)

        xi = np.arange(0, 3.0, 0.25)
        yi = function2(xi)
        u_yi = 0.03 * np.ones_like(yi)
        yi = cm.generate_sample(1, yi, u_yi, corr_x="rand").squeeze()
        x_HR = np.arange(0, 3.0, 0.09)
        y_HR = function2(x_HR)
        u_y_HR_syst = 0.9 * np.ones_like(y_HR)
        u_y_HR_rand = 0.02 * y_HR
        cov_y_HR = cm.convert_corr_to_cov(
            np.ones((len(y_HR), len(y_HR))), u_y_HR_syst
        ) + cm.convert_corr_to_cov(np.eye(len(y_HR)), u_y_HR_rand)
        corr_y_HR = cm.correlation_from_covariance(cov_y_HR)
        u_y_HR = cm.uncertainty_from_covariance(cov_y_HR)

        y_HR = cm.generate_sample(1, y_HR, u_y_HR, corr_x=corr_y_HR)

        xx = np.arange(0.1, 2.5, 0.02)

        y_hr_cubic = cm.interpolate_1d_along_example(
            xi,
            yi,
            x_HR,
            y_HR,
            xx,
            relative=True,
            method="cubic",
            method_hr="cubic",
        )

        y_hr_cubic2, u_y_hr_cubic2 = cm.interpolate_1d_along_example(
            xi,
            yi,
            x_HR,
            y_HR,
            xx,
            relative=True,
            method="cubic",
            method_hr="cubic",
            u_y_i=u_yi,
            corr_y_i="rand",
            u_y_hr=u_y_HR,
            corr_y_hr="syst",
            min_scale=0.3,
            return_uncertainties=True,
            plot_residuals=True,
            return_corr=False,
        )

        npt.assert_allclose(y_hr_cubic, y_hr_cubic2, atol=0.01)

        y_gpr, u_y_gpr = cm.interpolate_1d(
            xi,
            yi,
            xx,
            method="gpr",
            u_y_i=u_yi,
            min_scale=0.3,
            return_uncertainties=True,
        )
        y_hr_gpr = cm.interpolate_1d_along_example(
            xi,
            yi,
            x_HR,
            y_HR,
            xx,
            relative=False,
            method="gpr",
            method_hr="gpr",
            min_scale=0.3,
        )
        y_hr_gpr2, u_y_hr_gpr2 = cm.interpolate_1d_along_example(
            xi,
            yi,
            x_HR,
            y_HR,
            xx,
            relative=False,
            method="gpr",
            method_hr="gpr",
            u_y_i=u_yi,
            u_y_hr=u_y_HR,
            corr_y_i="rand",
            corr_y_hr=corr_y_HR,
            min_scale=0.3,
            return_uncertainties=True,
            plot_residuals=False,
            return_corr=False,
        )

        # npt.assert_allclose(y_hr_gpr,y_hr_gpr2,atol=0.01)

        mcprop = punpy.MCPropagation(50, parallel_cores=1)

        inp2 = cm.Interpolator(
            relative=False,
            method="gpr",
            method_hr="gpr",
            min_scale=0.3,
            add_model_error=True,
        )
        u_y_hr, corr2 = mcprop.propagate_random(
            inp2.interpolate_1d_along_example,
            [xi, yi, x_HR, y_HR, xx],
            [None, u_yi, None, u_y_HR, None],
            corr_x=[None, "rand", None, corr_y_HR, None],
            return_corr=True,
        )

        # npt.assert_allclose(u_y_hr_cubic2, u_y_hr, rtol=0.01)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xx, function2(xx), "b", label="True line")
        ax.plot(xi, yi, "ro", label="low-res data")
        ax.plot(x_HR, y_HR, "go", label="high-res data")
        ax.plot(
            xx,
            cm.interpolate_1d(xi, yi, xx, method="cubic"),
            "r:",
            label="cubic spline interpolation",
        )
        ax.plot(xx, y_gpr, "c:", label="GPR interpolation")
        ax.plot(xx, y_hr_gpr, "g-.", label="GPR interpolation with HR example")
        ax.plot(
            xx, y_hr_cubic, "m-.", label="cubic spline interpolation with HR example"
        )
        ax.fill_between(
            xx,
            y_hr_gpr2 - 1.9600 * u_y_hr,
            (y_hr_gpr2 + 1.9600 * u_y_hr),
            alpha=0.25,
            fc="c",
            ec="None",
            label="95% confidence interval",
            lw=0,
        )
        ax.fill_between(
            xx,
            y_hr_gpr2 - 1.9600 * u_y_hr_gpr2,
            (y_hr_gpr2 + 1.9600 * u_y_hr_gpr2),
            alpha=0.25,
            fc="g",
            ec="None",
            label="95% confidence interval",
            lw=0,
        )
        ax.fill_between(
            xx,
            y_hr_cubic2 - 1.9600 * u_y_hr_cubic2,
            (y_hr_cubic2 + 1.9600 * u_y_hr_cubic2),
            alpha=0.25,
            fc="m",
            ec="None",
            label="95% confidence interval",
            lw=0,
        )
        ax.legend(ncol=2, prop={"size": 6})
        ax.set_xlim([0.1, 2.5])
        # plt.show()
        fig.savefig("interpolation_test.pdf", bbox_inches="tight")

    def test_gaussian_process_regression(self):
        np.random.seed(1234567)

        xi = np.arange(0, 3.0, 0.25)
        yi = function2(xi)
        u_yi = 0.03 * np.ones_like(yi)
        yi = cm.generate_sample(1, yi, u_yi, corr_x="rand").squeeze()
        x_HR = np.arange(0, 3.0, 0.09)
        y_HR = function2(x_HR)
        u_y_HR_syst = 0.9 * np.ones_like(y_HR)
        u_y_HR_rand = 0.02 * y_HR
        cov_y_HR = cm.convert_corr_to_cov(
            np.ones((len(y_HR), len(y_HR))), u_y_HR_syst
        ) + cm.convert_corr_to_cov(np.eye(len(y_HR)), u_y_HR_rand)
        corr_y_HR = cm.correlation_from_covariance(cov_y_HR)
        u_y_HR = cm.uncertainty_from_covariance(cov_y_HR)

        y_HR = cm.generate_sample(1, y_HR, u_y_HR, corr_x=corr_y_HR)

        xx = np.arange(0.1, 2.5, 0.02)
        y_gpr, u_y_gpr = cm.gaussian_process_regression(
            xi, yi, xx, min_scale=0.3, return_uncertainties=True
        )

        y_gpr2, u_y_gpr2 = cm.gaussian_process_regression(
            xi,
            yi,
            xx,
            u_y_i=np.arange(len(u_yi)) * u_yi,
            min_scale=0.3,
            return_uncertainties=True,
        )

        y_gpr3, u_y_gpr3 = cm.gaussian_process_regression(
            xi,
            yi,
            xx,
            u_y_i=np.arange(len(u_yi)) * u_yi,
            min_scale=0.3,
            return_uncertainties=True,
            include_model_uncertainties=False,
        )

        u_y_gpr = (u_y_gpr**2 + (u_y_gpr3) ** 2) ** 0.5

        y_hr_gpr, u_y_hr_gpr = cm.gaussian_process_regression(
            x_HR,
            y_HR,
            xx,
            u_y_i=u_y_HR,
            corr_y_i=corr_y_HR,
            min_scale=0.3,
            return_uncertainties=True,
        )

        print(u_y_hr_gpr, u_y_HR)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xx, function2(xx), "b", label="True line")
        ax.plot(xi, yi, "ro", label="low-res data")
        ax.plot(x_HR, y_HR, "go", label="high-res data")
        ax.plot(xx, y_hr_gpr, "g-.", label="GPR interpolation HR example")
        ax.fill_between(
            xx,
            y_hr_gpr - 1.9600 * u_y_hr_gpr,
            (y_hr_gpr + 1.9600 * u_y_hr_gpr),
            alpha=0.25,
            fc="g",
            ec="None",
            label="95% confidence interval",
            lw=0,
        )
        ax.plot(xx, y_gpr, "r-", label="GPR interpolation example")
        ax.fill_between(
            xx,
            y_gpr - 1.9600 * u_y_gpr,
            (y_gpr + 1.9600 * u_y_gpr),
            alpha=0.25,
            fc="r",
            ec="None",
            label="95% confidence interval uncertainties added separately",
            lw=0,
        )
        ax.plot(xx, y_gpr2, "m-", label="GPR interpolation example with combined unc")
        ax.fill_between(
            xx,
            y_gpr2 - 1.9600 * u_y_gpr2,
            (y_gpr2 + 1.9600 * u_y_gpr2),
            alpha=0.25,
            fc="m",
            ec="None",
            label="95% confidence interval",
            lw=0,
        )
        ax.legend(ncol=2, prop={"size": 6})
        ax.set_xlim([0.1, 2.5])
        fig.savefig("interpolation_gpr.png")

    def test_extrapolation(self):
        xi = np.arange(0, 2.8, 0.25)
        yi = function2(xi)
        u_yi = 0.03 * np.ones_like(yi)
        yi = cm.generate_sample(1, yi, u_yi, corr_x="rand").squeeze()
        x_HR = np.arange(-0.5, 4.0, 0.09)
        y_HR = function2(x_HR)
        u_y_HR_syst = 0.9 * np.ones_like(y_HR)
        u_y_HR_rand = 0.02 * y_HR
        cov_y_HR = cm.convert_corr_to_cov(
            np.ones((len(y_HR), len(y_HR))), u_y_HR_syst
        ) + cm.convert_corr_to_cov(np.eye(len(y_HR)), u_y_HR_rand)
        corr_y_HR = cm.correlation_from_covariance(cov_y_HR)
        u_y_HR = cm.uncertainty_from_covariance(cov_y_HR)

        y_HR = cm.generate_sample(1, y_HR, u_y_HR, corr_x=corr_y_HR)

        xx2 = np.arange(0.0, 3.5, 0.02)
        y_hr_gpr3, u_y_hr_gpr3 = cm.interpolate_1d_along_example(
            xi,
            yi,
            x_HR,
            y_HR,
            xx2,
            relative=False,
            method="gpr",
            method_hr="gpr",
            u_y_i=u_yi,
            u_y_hr=u_y_HR,
            corr_y_i="rand",
            corr_y_hr=corr_y_HR,
            min_scale=0.3,
            extrapolate="nearest",
            return_uncertainties=True,
            plot_residuals=False,
            return_corr=False,
        )

        y_hr_lin, u_y_hr_lin = cm.interpolate_1d(
            xi,
            yi,
            xx2,
            method="lagrange",
            u_y_i=u_yi,
            corr_y_i="rand",
            extrapolate="extrapolate",
            return_uncertainties=True,
            return_corr=False,
        )

        y_hr_lin2, u_y_hr_lin2 = cm.interpolate_1d(
            xi,
            yi,
            xx2,
            method="gpr",
            u_y_i=u_yi,
            corr_y_i="rand",
            extrapolate="linear",
            return_uncertainties=True,
            return_corr=False,
        )

        fig3 = plt.figure(figsize=(10, 5))
        ax = fig3.add_subplot(1, 1, 1)
        ax.plot(xx2, function2(xx2), "b", label="True line")
        ax.plot(xi, yi, "ro", label="low-res data")
        ax.plot(x_HR, y_HR, "go", label="high-res data")
        ax.plot(xx2, y_hr_lin, "b--", label="linear interpolation with extrapolation")
        ax.fill_between(
            xx2,
            y_hr_lin - 1.9600 * u_y_hr_lin,
            (y_hr_lin + 1.9600 * u_y_hr_lin),
            alpha=0.15,
            fc="b",
            ec="None",
            lw=0,
        )
        ax.plot(
            xx2, y_hr_lin2, "c--", label="gpr interpolation with linear extrapolation"
        )
        ax.fill_between(
            xx2,
            y_hr_lin2 - 1.9600 * u_y_hr_lin2,
            (y_hr_lin2 + 1.9600 * u_y_hr_lin2),
            alpha=0.15,
            fc="c",
            ec="None",
            lw=0,
        )
        ax.plot(
            xx2,
            y_hr_gpr3,
            "g--",
            label="GPR interpolation with HR example and extrapolation",
        )
        ax.fill_between(
            xx2,
            y_hr_gpr3 - 1.9600 * u_y_hr_gpr3,
            (y_hr_gpr3 + 1.9600 * u_y_hr_gpr3),
            alpha=0.15,
            fc="g",
            ec="None",
            lw=0,
        )
        ax.set_ylim(-5, 5)
        ax.set_xlim(0, 3.5)
        ax.legend(ncol=2, prop={"size": 8})
        fig3.savefig("test_extrapolation.png")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unittest.main()
