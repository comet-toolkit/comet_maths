from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from scipy.interpolate import lagrange

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
import punpy
import comet_maths as cm


class Interpolator:
    def __init__(
        self,
        relative=True,
        method="cubic",
        method_hr="cubic",
        unc_methods=None,
        add_model_error=True,
        min_scale=0.3,
        extrapolate="nearest",
        plot_residuals=False,
    ):
        self.relative = relative
        self.method = method
        self.method_hr = method_hr
        self.add_model_error = add_model_error
        self.min_scale = min_scale
        self.extrapolate = extrapolate
        self.plot_residuals = plot_residuals
        self.unc_methods = unc_methods

    def interpolate_1d_along_example(self, x_i, y_i, x_hr, y_hr, x):
        """
        Method for interpolating between datapoints by following an example.
        The example can come from either models or higher-resolution observations.
        Here the example is assumed to have an unknown normalisation or poor absolute calibration,
        yet the low resolution data has a more precise calibration (and can thus be used to constrain the high-resolution model).

        :param x_i: Independent variable quantity x for the low resolution data
        :type x_i: ndarray
        :param y_i: measured variable quantity y for the low resolution data
        :type y_i: ndarray
        :param x_hr: Independent variable quantity x for the high resolution data
        :type x_hr: ndarray
        :param y_hr: measured variable quantity y for the high resolution data
        :type y_hr: ndarray
        :param x: Independent variable quantity x for which we are trying to obtain the measurand y
        :type x: ndarray
        :return: The measurand y evaluated at the values x
        :rtype: ndarray
        """
        return interpolate_1d_along_example(
            x_i,
            y_i,
            x_hr,
            y_hr,
            x,
            relative=self.relative,
            u_y_i=None,
            corr_y_i=None,
            u_y_hr=None,
            corr_y_hr=None,
            min_scale=self.min_scale,
            method_hr=self.method_hr,
            method=self.method,
            return_uncertainties=False,
            return_corr=False,
            add_model_error=self.add_model_error,
            extrapolate=self.extrapolate,
            plot_residuals=self.plot_residuals,
        )

    def interpolate_1d(self, x_i, y_i, x):
        """
        Interpolates 1D data to defined coordinates x in 1D

        :param x_i: initial coordinate data of y_i
        :type x_i: numpy.ndarray
        :param y_i: data to interpolate
        :type y_i: numpy.ndarray
        :param x: coordinate data to interpolate y_i to
        :type x: numpy.ndarray
        :return: interpolate data
        :rtype: numpy.ndarray
        """

        return interpolate_1d(
            x_i,
            y_i,
            x,
            min_scale=self.min_scale,
            method=self.method,
            unc_methods=self.unc_methods,
            return_uncertainties=False,
            add_model_error=self.add_model_error,
            extrapolate=self.extrapolate,
        )


def interpolate(
    x_i, y_i, x, method="linear", return_uncertainties=False, add_model_error=False
):
    """
    Interpolates data to defined coordinates x

    :param x_i: initial coordinate data of y_i
    :type x_i: numpy.ndarray
    :param y_i: data to interpolate
    :type y_i: numpy.ndarray
    :param x: coordinate data to interpolate y_i to
    :type x: numpy.ndarray
    :param method: interpolation method to be used, defaults to linear
    :type method: string (optional)
    :param return_uncertainties: Boolean to indicate whether interpolation (model error) uncertainties should be calculated and returned. Defaults to False
    :type return_uncertainties: bool (optional)
    :param add_model_error: Boolean to indicate whether model error should be added to account for interpolation uncertainties (useful in Monte Carlo approaches). Defaults to False
    :type add_model_error: bool (optional)
    :return: interpolate data
    :rtype: numpy.ndarray
    """
    x_i = np.array(x_i)
    y_i = np.array(y_i)

    y_i = y_i[~np.isnan(x_i)]
    x_i = x_i[~np.isnan(x_i)]

    x_i = x_i[~np.isnan(y_i)]
    y_i = y_i[~np.isnan(y_i)]

    if x_i.ndim == 1:
        interpolate_1d(x_i, y_i, x, method, return_uncertainties, add_model_error)


def interpolate_1d(
    x_i,
    y_i,
    x,
    method="cubic",
    u_y_i=None,
    corr_y_i=None,
    unc_methods=None,
    min_scale=0.3,
    return_uncertainties=False,
    add_model_error=False,
    return_corr=False,
    extrapolate="nearest",
    MCsteps=100,
    parallel_cores=4,
):
    """
    Interpolates 1D data to defined coordinates x in 1D

    :param x_i: initial coordinate data of y_i
    :type x_i: numpy.ndarray
    :param y_i: data to interpolate
    :type y_i: numpy.ndarray
    :param x: coordinate data to interpolate y_i to
    :type x: numpy.ndarray
    :param method: interpolation method to be used, defaults to linear
    :type method: string (optional)
    :param return_uncertainties: Boolean to indicate whether interpolation (model error) uncertainties should be calculated and returned. Defaults to False
    :type return_uncertainties: bool (optional)
    :param add_model_error: Boolean to indicate whether model error should be added to account for interpolation uncertainties (useful in Monte Carlo approaches). Defaults to False
    :type add_model_error: bool (optional)
    :return: interpolate data
    :rtype: numpy.ndarray
    """
    if method.lower() == "gpr":
        return gaussian_progress_regression(
            x_i,
            y_i,
            x,
            min_scale=min_scale,
            u_y_i=u_y_i,
            corr_y_i=corr_y_i,
            return_uncertainties=return_uncertainties,
            add_model_error=add_model_error,
            return_corr=return_corr,
            MCsteps=MCsteps,
            parallel_cores=parallel_cores,
        )

    if unc_methods is None:
        if method.lower() == "nearest":
            unc_methods = ["nearest", "previous", "next"]
        elif method.lower() == "linear":
            unc_methods = ["linear", "quadratic", "cubic"]
        elif method.lower() == "quadratic":
            unc_methods = ["linear", "quadratic", "cubic"]
        elif method.lower() == "cubic":
            unc_methods = ["linear", "quadratic", "cubic"]
        elif method.lower() == "lagrange":
            unc_methods = ["lagrange", "quadratic", "cubic"]
        else:
            raise ValueError(
                "comet_maths.interpolation: uncertainties for the model error for this interpolation method (%s) are not yet implemented"
                % (method)
            )

    if add_model_error:
        method = np.random.choice(unc_methods)
        extrapolate = np.random.choice(["nearest","extrapolate"])

    if method.lower() in [
        "linear",
        "nearest",
        "nearest-up",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "previous",
        "next",
    ]:
        if extrapolate == "nearest":
            f_i = interp1d(
                x_i,
                y_i,
                kind=method.lower(),
                fill_value=(y_i[0], y_i[-1]),
                bounds_error=False,
            )
        elif extrapolate == "extrapolate":
            f_i = interp1d(x_i, y_i, kind=method.lower(), fill_value="extrapolate")
        else:
            f_i = interp1d(x_i, y_i, kind=method.lower())

        y = f_i(x).squeeze()

    elif method.lower() == "ius":
        f_i = InterpolatedUnivariateSpline(x_i, y_i)
        y = f_i(x).squeeze()

    elif method.lower() == "lagrange":
        f_i = lagrange(x_i, y_i)
        y = f_i(x).squeeze()

    else:
        raise ValueError(
            "comet_maths.interpolation: this interpolation method (%s) is not implemented"
            % (method)
        )

    if return_uncertainties or return_corr:

        y_unc, y_corr = unc_interpolate_1d(
            x_i,
            y_i,
            x,
            u_y_i=u_y_i,
            corr_y_i=corr_y_i,
            unc_methods=unc_methods,
            MCsteps=MCsteps,
            parallel_cores=parallel_cores,
        )
        y_corr[np.where(np.isnan(y_corr))] = 0

    if return_uncertainties:
        if return_corr:
            return y, y_unc, y_corr
        else:
            return y, y_unc

    else:
        return y


def unc_interpolate_1d(
    x_i,
    y_i,
    x,
    unc_methods=None,
    u_y_i=None,
    corr_y_i=None,
    MCsteps=100,
    parallel_cores=4,
):
    if u_y_i is None:
        u_y, corr_y = std_interpolation_methods(x_i, y_i, x, methods=unc_methods)
    else:
        prop = punpy.MCPropagation(MCsteps, parallel_cores=parallel_cores)
        intp = Interpolator(unc_methods=unc_methods, add_model_error=True)
        u_y, corr_y = prop.propagate_random(
            intp.interpolate_1d,
            [x_i, y_i, x],
            [None, u_y_i, None],
            corr_x=[None, corr_y_i, None],
            return_corr=True,
        )
    return u_y, corr_y


def std_interpolation_methods(x_i, y_i, x, methods=["linear", "quadratic", "cubic"]):
    data = np.zeros((len(x), len(methods)))
    for i in range(len(methods)):
        data[:, i] = interpolate_1d(x_i, y_i, x, methods[i])
    return np.std(data, axis=1), np.corrcoef(data)


def gaussian_progress_regression(
    x_i,
    y_i,
    x,
    u_y_i=None,
    corr_y_i=None,
    kernel="RBF",
    min_scale=0.01,
    max_scale=10000,
    return_uncertainties=True,
    add_model_error=False,
    return_corr=False,
    MCsteps=100,
    parallel_cores=4,
):
    X = np.atleast_2d(x_i).T

    # Observations
    y = np.atleast_2d(y_i).T

    if corr_y_i is None:
        if u_y_i is None:
            alpha = 1e-10  # default value for GaussianProcessRegressor
        else:
            alpha = u_y_i ** 2

        # Mesh the input space for evaluations of the real function, the prediction and
        # its MSE
        x = np.atleast_2d(x).T

        # Instantiate a Gaussian Process model
        if kernel == "RBF":
            kernel_mod = C(1.0, (1e-9, 1e9)) * RBF(
                length_scale=0.3, length_scale_bounds=(min_scale, max_scale)
            )
        if kernel == "exp":
            kernel_mod = C(1.0, (1e-9, 1e9)) * Matern(
                length_scale=0.3, length_scale_bounds=(min_scale, max_scale), nu=0.5
            )
        gp = GaussianProcessRegressor(
            kernel=kernel_mod, alpha=alpha, n_restarts_optimizer=9
        )

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(X, y)

        # print("KERNEL",gp.score(X,y),gp2.score(X,y))
        # print("KERNEL2",gp.log_marginal_likelihood(),gp2.log_marginal_likelihood())
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred, u_y_out = gp.predict(x, return_std=True)
        y_pred, cov = gp.predict(x, return_cov=True)
        y_out = y_pred.squeeze()
        corr_y_out = cm.correlation_from_covariance(cov)
        #print(y_unc, cm.uncertainty_from_covariance(cov))

        if add_model_error:
            y_out = cm.generate_sample(1, y_out, u_y_out, corr_y_out).squeeze()

    else:
        if corr_y_i=="rand":
            return gaussian_progress_regression(
                x_i,
                y_i,
                x,
                u_y_i=u_y_i,
                corr_y_i=None,
                kernel=kernel,
                min_scale=min_scale,
                max_scale=max_scale,
                return_uncertainties=return_uncertainties,
                add_model_error=add_model_error,
                return_corr=return_corr,
                MCsteps=MCsteps,
                parallel_cores=parallel_cores,
            )

        elif isinstance(corr_y_i,np.ndarray):
            if corr_y_i==np.diag(corr_y_i):
                return gaussian_progress_regression(
                    x_i,
                    y_i,
                    x,
                    u_y_i=u_y_i,
                    corr_y_i=None,
                    kernel=kernel,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    return_uncertainties=return_uncertainties,
                    add_model_error=add_model_error,
                    return_corr=return_corr,
                    MCsteps=MCsteps,
                    parallel_cores=parallel_cores,
                )

        else:
            prop = punpy.MCPropagation(MCsteps, parallel_cores=parallel_cores)
            intp = Interpolator(method="gpr", min_scale=min_scale, add_model_error=True)
            y_out = gaussian_progress_regression(x_i,
                                                 y_i,
                                                 x,
                                                 kernel=kernel,
                                                 min_scale=min_scale,
                                                 max_scale=max_scale,
                                                 return_uncertainties=False,
                                                 add_model_error=False,
                                                 return_corr=False)
            u_y_out, corr_y_out = prop.propagate_random(
                intp.interpolate_1d,
                [x_i, y_i, x],
                [None, u_y_i, None],
                corr_x=[None, corr_y_i, None],
                return_corr=True,
            )

    if return_uncertainties:
        if return_corr:
            return y_out, u_y_out, corr_y_out
        else:
            return y_out, u_y_out

    else:
        return y_out


def interpolate_1d_along_example(
    x_i,
    y_i,
    x_hr,
    y_hr,
    x,
    relative=True,
    u_y_i=None,
    corr_y_i=None,
    u_y_hr=None,
    corr_y_hr=None,
    min_scale=0.3,
    method="gpr",
    method_hr="cubic",
    unc_methods=None,
    unc_methods_hr=None,
    return_uncertainties=False,
    return_corr=False,
    add_model_error=False,
    extrapolate="nearest",
    plot_residuals=False,
    MCsteps=100,
    parallel_cores=4,
):
    """
    Method for interpolating between datapoints by following an example.
    The example can come from either models or higher-resolution observations.
    Here the example is assumed to have an unknown normalisation or poor absolute calibration,
    yet the low resolution data has a more precise calibration (and can thus be used to constrain the high-resolution model).

    :param x_i: Independent variable quantity x for the low resolution data
    :type x_i: ndarray
    :param y_i: measured variable quantity y for the low resolution data
    :type y_i: ndarray
    :param x_hr: Independent variable quantity x for the high resolution data
    :type x_hr: ndarray
    :param y_hr: measured variable quantity y for the high resolution data
    :type y_hr: ndarray
    :param x: Independent variable quantity x for which we are trying to obtain the measurand y
    :type x: ndarray
    :param relative: Boolean to indicate whether a relative normalisation (True) or absolute normalisation (False) should be used. Defaults to True.
    :type relative: bool (optional)
    :param method: Sting to indicate which interpolation method should be used to interpolate between normalised data (core interpolation step within the approach). Defaults to Gaussian Progress Regression.
    :type method: string (optional)
    :param method_hr: String to indicate which interpolation method should be used to interpolate between high resolution measurements. Defaults to cubic spline interpolation.
    :type method_hr: string (optional)
    :param return_uncertainties: Boolean to indicate whether interpolation uncertainties should be returned
    :type return_uncertainties: bool (optional)
    :param add_model_error: Boolean to indicate whether errors should be added to account for interpolation uncertainty (useful for MC). Defaults to False
    :type add_model_error: bool (optional)
    :return: The measurand y evaluated at the values x
    :rtype: ndarray
    """
    if not return_corr:
        if return_uncertainties:
            y_hr_i2 = interpolate_1d(
                x_hr,
                y_hr,
                x_i,
                method=method_hr,
                min_scale=min_scale,
                extrapolate=extrapolate,)


            y_hr_i, u_y_hr_i = interpolate_1d(
                x_hr,
                y_hr,
                x_i,
                method=method_hr,
                u_y_i=u_y_hr,
                corr_y_i=corr_y_hr,
                unc_methods=unc_methods_hr,
                min_scale=min_scale,
                return_uncertainties=return_uncertainties,
                return_corr=return_corr,
                add_model_error=add_model_error,
                extrapolate=extrapolate,
                MCsteps=MCsteps,
                parallel_cores=parallel_cores,
            )
            print("yy",y_hr_i2,y_hr_i)

            if x == x_hr:
                y_hr_out = y_hr
            else:
                y_hr_out, u_y_hr_out = interpolate_1d(
                    x_hr,
                    y_hr,
                    x,
                    method=method_hr,
                    u_y_i=u_y_hr,
                    corr_y_i=corr_y_hr,
                    unc_methods=unc_methods_hr,
                    min_scale=min_scale,
                    return_uncertainties=return_uncertainties,
                    add_model_error=add_model_error,
                    extrapolate=extrapolate,
                    MCsteps=MCsteps,
                    parallel_cores=parallel_cores,
                )

            if x == x_hr:
                y_hr_out2 = y_hr
            else:
                y_hr_out2 = interpolate_1d(
                    x_hr,
                    y_hr,
                    x,
                    method=method_hr,
                    min_scale=min_scale,
                    extrapolate=extrapolate,
                )
            if relative:
                y_norm_i = y_i / y_hr_i
                u_y_norm_i = (
                    (u_y_i / y_i) ** 2 + (u_y_hr_i / y_hr_i) ** 2
                ) ** 0.5 * y_norm_i
                u_y_norm_i[np.where(y_norm_i==0.)]=0.
            else:
                y_norm_i = y_i - y_hr_i
                u_y_norm_i = ((u_y_i) ** 2 + (u_y_hr_i) ** 2) ** 0.5

            y_norm_hr, u_y_norm_hr = interpolate_1d(
                x_i,
                y_norm_i,
                x,
                method=method,
                u_y_i=u_y_norm_i,
                corr_y_i=corr_y_i,
                min_scale=min_scale,
                return_uncertainties=return_uncertainties,
                unc_methods=unc_methods,
                extrapolate=extrapolate,
                MCsteps=MCsteps,
                parallel_cores=parallel_cores,
            )
            y_norm_hr = y_norm_hr.squeeze()

            if relative:
                r = -u_y_hr_out / u_y_norm_hr

                y_out = y_norm_hr * y_hr_out
                var=(
                        u_y_hr_out / y_hr_out
                        + 2 * r * u_y_norm_hr / y_norm_hr * u_y_hr_out / y_hr_out
                )** 2
                u_y_out = (
                    (u_y_norm_hr / y_norm_hr) ** 2
                    + (
                        u_y_hr_out / y_hr_out
                        + 2 * r * u_y_norm_hr / y_norm_hr * u_y_hr_out / y_hr_out
                    )
                    ** 2
                ) ** 0.5 * y_out
            else:
                r = -u_y_hr_out / u_y_norm_hr
                y_out = y_norm_hr + y_hr_out
                u_y_out = (
                    (u_y_norm_hr) ** 2
                    + (u_y_hr_out) ** 2
                    + 2 * r * u_y_norm_hr * u_y_hr_out
                ) ** 0.5

            if plot_residuals:
                plt.plot(x_i, y_norm_i, "ro", label="low-res residuals")
                plt.plot(x, y_norm_hr, "g-", label="high-res residuals")
                plt.fill_between(
                    x,
                    y_norm_hr - 1.9600 * u_y_norm_hr,
                    (y_norm_hr + 1.9600 * u_y_norm_hr),
                    alpha=0.25,
                    fc="g",
                    ec="None",
                    label="95% confidence interval",
                    lw=0,
                )

                plt.ylabel("Residuals")
                plt.xlabel("x")
                plt.legend()
                plt.savefig("residuals.png")
                plt.clf()

        else:
            y_hr_i = interpolate_1d(
                x_hr,
                y_hr,
                x_i,
                method=method_hr,
                min_scale=min_scale,
                extrapolate=extrapolate,
            )

            if x == x_hr:
                y_hr_out = y_hr
            else:
                y_hr_out = interpolate_1d(
                    x_hr,
                    y_hr,
                    x,
                    method=method_hr,
                    min_scale=min_scale,
                    extrapolate=extrapolate,
                )

            if relative:
                y_norm_i = y_i / y_hr_i
            else:
                y_norm_i = y_i - y_hr_i

            y_norm_hr = interpolate_1d(
                x_i,
                y_norm_i,
                x,
                method=method,
                min_scale=min_scale,
                extrapolate=extrapolate,
            )

            if relative:
                y_out = y_norm_hr * y_hr_out
            else:
                y_out = y_norm_hr + y_hr_out

            if plot_residuals:
                plt.plot(x_i, y_norm_i, "ro", label="low-res residuals")
                plt.plot(x, y_norm_hr, "g-", label="high-res residuals")
                plt.ylabel("Residuals")
                plt.xlabel("x")
                plt.legend()
                plt.savefig("residuals.png")
                plt.clf()

    else:
        prop = punpy.MCPropagation(MCsteps, parallel_cores=parallel_cores)
        intp = Interpolator(
            method=method,
            method_hr=method_hr,
            relative=relative,
            min_scale=min_scale,
            extrapolate=extrapolate,
            add_model_error=True,
        )
        u_y_out, corr_y_out = prop.propagate_random(
            intp.interpolate_1d_along_example,
            [x_i, y_i, x_hr, y_hr, x],
            [None, u_y_i, None, u_y_hr, None],
            corr_x=[None, corr_y_i, None, corr_y_hr, None],
            return_corr=True,
        )
        y_out = interpolate_1d_along_example(
            x_i,
            y_i,
            x_hr,
            y_hr,
            x,
            relative=relative,
            u_y_i=None,
            corr_y_i=None,
            u_y_hr=None,
            corr_y_hr=None,
            min_scale=min_scale,
            method=method,
            method_hr=method_hr,
            unc_methods=None,
            unc_methods_hr=None,
            return_uncertainties=False,
            return_corr=False,
            add_model_error=False,
            extrapolate=extrapolate,
            plot_residuals=plot_residuals,
        )

    if return_uncertainties:
        if return_corr:
            return y_out, u_y_out, corr_y_out
        else:
            return y_out, u_y_out

    else:
        return y_out
