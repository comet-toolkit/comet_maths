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
    x_i, y_i, x, method="linear", return_uncertainties=False, add_model_error=False, include_model_uncertainties=True,
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
        interpolate_1d(x_i, y_i, x, method, return_uncertainties=return_uncertainties, add_model_error=add_model_error, include_model_uncertainties=include_model_uncertainties)

    else:
        raise NotImplementedError()

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
    include_model_uncertainties=True,
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

    if (not return_corr) and (not return_uncertainties) and (not add_model_error):
        return y

    else:
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
                unc_methods = ["lagrange", "linear", "cubic"]
            elif method.lower() == "ius":
                unc_methods = ["ius", "linear", "cubic"]
            else:
                raise ValueError(
                    "comet_maths.interpolation: uncertainties for the model error for this interpolation method (%s) are not yet implemented"
                    % (method)
                )

        if add_model_error or include_model_uncertainties:
            u_y_model, corr_y_model, cov_model = std_interpolation_methods(x_i, y_i, x, methods=unc_methods)

        if add_model_error:
            #y=cm.generate_sample(1,y,u_y_model,corr_y_model)
            y = cm.generate_sample_cov(1,y,cov_model,diff=0.1).squeeze()

        if (not return_uncertainties) and (not return_corr):
            return y

        if u_y_i is None:
            y_unc = u_y_model
            y_corr = corr_y_model

        else:
            prop = punpy.MCPropagation(MCsteps, parallel_cores=1)
            intp = Interpolator(unc_methods=unc_methods, add_model_error=include_model_uncertainties)
            y_unc, y_corr = prop.propagate_random(
                intp.interpolate_1d,
                [x_i, y_i, x],
                [None, u_y_i, None],
                corr_x=[None, corr_y_i, None],
                return_corr=True,
            )
            y_corr[np.where(np.isnan(y_corr))] = 0

        if return_uncertainties and return_corr:
            return y, y_unc, y_corr
        else:
            return y, y_unc


def std_interpolation_methods(x_i, y_i, x, methods=["linear", "quadratic", "cubic"]):
    data = np.zeros((len(x), len(methods)))
    for i in range(len(methods)):
        data[:, i] = interpolate_1d(x_i, y_i, x, methods[i])
    return np.std(data, axis=1), np.corrcoef(data), np.cov(data)


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
    include_model_uncertainties=True,
    return_corr=False,
    MCsteps=100,
    parallel_cores=4,
):
    # First calculate y_out without uncertainties
    y_out, cov_model = gpr_basics(x_i,
                       y_i,
                       x,
                       kernel=kernel,
                       min_scale=min_scale,
                       max_scale=max_scale)

    if add_model_error:
        y_out = cm.generate_sample_cov(1,y_out,cov_model,diff=0.1).squeeze()

    if (not return_uncertainties) and (not return_corr):
        return y_out

    if (u_y_i is None) and include_model_uncertainties:
        corr_y_out = cm.correlation_from_covariance(cov_model)
        u_y_out = cm.uncertainty_from_covariance(cov_model)

    else:
        #next determine if a simple uncertainty from gpr is possible or if MC is necessary
        uncertainties_simple=False
        if (u_y_i is not None) and include_model_uncertainties:
            if (corr_y_i is None) or (corr_y_i=="rand"):
                uncertainties_simple=True
            elif isinstance(corr_y_i,np.ndarray):
                if np.all(corr_y_i==np.diag(corr_y_i)):
                    uncertainties_simple=True

        if uncertainties_simple:
            y_out_simple, cov_simple = gpr_basics(x_i,
                                        y_i,
                                        x,
                                        u_y_i=u_y_i,
                                        kernel=kernel,
                                        min_scale=min_scale,
                                        max_scale=max_scale)
            corr_y_out = cm.correlation_from_covariance(cov_simple)
            u_y_out = cm.uncertainty_from_covariance(cov_simple)

        else:
            prop = punpy.MCPropagation(MCsteps, parallel_cores=1)
            intp = Interpolator(method="gpr", min_scale=min_scale, add_model_error=include_model_uncertainties)
            u_y_out, corr_y_out = prop.propagate_random(
                intp.interpolate_1d,
                [x_i, y_i, x],
                [None, u_y_i, None],
                corr_x=[None, corr_y_i, None],
                return_corr=True,
            )

    if return_uncertainties and return_corr:
        return y_out, u_y_out, corr_y_out
    else:
        return y_out, u_y_out



def gpr_basics(x_i, y_i, x, u_y_i=None, kernel="RBF",
               min_scale=0.01,
               max_scale=10000,):
    """

    :param x_i:
    :type x_i:
    :param y_i:
    :type y_i:
    :param x:
    :type x:
    :param u_y_i:
    :type u_y_i:
    :param return_cov:
    :type return_cov:
    :param kernel:
    :type kernel:
    :param min_scale:
    :type min_scale:
    :param max_scale:
    :type max_scale:
    :return:
    :rtype:
    """
    X = np.atleast_2d(x_i).T

    # Observations
    y = np.atleast_2d(y_i).T

    if u_y_i is None:
        alpha = 1e-10  # default value for GaussianProcessRegressor
    else:
        alpha = u_y_i ** 2

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    xt = np.atleast_2d(x).T

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


    y_pred, cov = gp.predict(xt, return_cov=True)
    y_out = y_pred.squeeze()
    return y_out, cov


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
    include_model_uncertainties=True,
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
    y_hr_i = interpolate_1d(
        x_hr,
        y_hr,
        x_i,
        method=method_hr,
        min_scale=min_scale,
        extrapolate=extrapolate,
        add_model_error=add_model_error
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
            add_model_error=add_model_error
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
        add_model_error=add_model_error
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

    if (not return_corr) and (not return_uncertainties):
        return y_out

    # if return_uncertainties:
        #     y_hr_i, u_y_hr_i, corr_y_hr_i = interpolate_1d(
        #         x_hr,
        #         y_hr,
        #         x_i,
        #         method=method_hr,
        #         u_y_i=u_y_hr,
        #         corr_y_i=corr_y_hr,
        #         unc_methods=unc_methods_hr,
        #         min_scale=min_scale,
        #         return_uncertainties=return_uncertainties,
        #         return_corr=True,
        #         add_model_error=add_model_error,
        #         include_model_uncertainties=include_model_uncertainties,
        #         extrapolate=extrapolate,
        #         MCsteps=MCsteps,
        #         parallel_cores=parallel_cores,
        #     )
        #     cov_y_hr_i = cm.convert_corr_to_cov(corr_y_hr_i, u_y_hr_i)
        #     if x == x_hr:
        #         y_hr_out = y_hr
        #     else:
        #         y_hr_out, u_y_hr_out, corr_y_hr_out = interpolate_1d(
        #             x_hr,
        #             y_hr,
        #             x,
        #             method=method_hr,
        #             u_y_i=u_y_hr,
        #             corr_y_i=corr_y_hr,
        #             unc_methods=unc_methods_hr,
        #             min_scale=min_scale,
        #             return_uncertainties=return_uncertainties,
        #             return_corr=True,
        #             add_model_error=add_model_error,
        #             include_model_uncertainties=include_model_uncertainties,
        #             extrapolate=extrapolate,
        #             MCsteps=MCsteps,
        #             parallel_cores=parallel_cores,
        #         )
        #         cov_y_hr_out = cm.convert_corr_to_cov(corr_y_hr_out, u_y_hr_out)
        #
        # # if x == x_hr:
        #     #     y_hr_out2 = y_hr
        #     # else:
        #     #     y_hr_out2 = interpolate_1d(
        #     #         x_hr,
        #     #         y_hr,
        #     #         x,
        #     #         method=method_hr,
        #     #         min_scale=min_scale,
        #     #         extrapolate=extrapolate,
        #     #         include_model_uncertainties=include_model_uncertainties,
        #     #     )
        #     if relative:
        #         y_norm_i = y_i / y_hr_i
        #         u_y_norm_i = (
        #             (u_y_i / y_i) ** 2 + (u_y_hr_i / y_hr_i) ** 2
        #         ) ** 0.5 * y_norm_i
        #         u_y_norm_i[np.where(y_norm_i==0.)]=0.
        #     else:
        #         y_norm_i = y_i - y_hr_i
        #         u_y_norm_i = ((u_y_i) ** 2 + (u_y_hr_i) ** 2) ** 0.5
        #         if corr_y_i=="rand":
        #             corr_y_i=np.eye(len(u_y_i))
        #         if corr_y_i=="syst":
        #             corr_y_i=np.ones((len(u_y_i),len(y_i)))
        #
        #         cov_y_i = cm.convert_corr_to_cov(corr_y_i, u_y_i)
        #
        #         cov_y_norm_i = cov_y_i + cov_y_hr_i
        #         u_y_norm_i2 = cm.uncertainty_from_covariance(cov_y_norm_i)
        #         corr_y_norm_i = cm.correlation_from_covariance(cov_y_norm_i)
        #
        #     y_norm_hr, u_y_norm_hr = interpolate_1d(
        #         x_i,
        #         y_norm_i,
        #         x,
        #         method=method,
        #         u_y_i=u_y_norm_i,
        #         corr_y_i=corr_y_norm_i,
        #         min_scale=min_scale,
        #         return_uncertainties=return_uncertainties,
        #         unc_methods=unc_methods,
        #         extrapolate=extrapolate,
        #         MCsteps=MCsteps,
        #         parallel_cores=parallel_cores,
        #         include_model_uncertainties=include_model_uncertainties,
        #     )
        #     y_norm_hr = y_norm_hr.squeeze()
        #
        #     if relative:
        #         r = -u_y_hr_out / u_y_norm_hr
        #         y_out = y_norm_hr * y_hr_out
        #         var=(
        #                 u_y_hr_out / y_hr_out
        #                 + 2 * r * u_y_norm_hr / y_norm_hr * u_y_hr_out / y_hr_out
        #         )** 2
        #         u_y_out = (
        #             (u_y_norm_hr / y_norm_hr) ** 2
        #             + (
        #                 u_y_hr_out / y_hr_out
        #                 + 2 * r * u_y_norm_hr / y_norm_hr * u_y_hr_out / y_hr_out
        #             )
        #             ** 2
        #         ) ** 0.5 * y_out
        #     else:
        #         r = -u_y_hr_out / u_y_norm_hr
        #         y_out = y_norm_hr + y_hr_out
        #         u_y_out = (
        #             (u_y_norm_hr) ** 2
        #             - (u_y_hr_out) ** 2
        #
        #         ) ** 0.5
        #
        #     if plot_residuals:
        #         plt.plot(x_i, y_norm_i, "ro", label="low-res residuals")
        #         plt.plot(x, y_norm_hr, "g-", label="high-res residuals")
        #         plt.fill_between(
        #             x,
        #             y_norm_hr - 1.9600 * u_y_norm_hr,
        #             (y_norm_hr + 1.9600 * u_y_norm_hr),
        #             alpha=0.25,
        #             fc="g",
        #             ec="None",
        #             label="95% confidence interval",
        #             lw=0,
        #         )
        #
        #         plt.ylabel("Residuals")
        #         plt.xlabel("x")
        #         plt.legend()
        #         plt.savefig("residuals.png")
        #         plt.clf()
        #
        # else:


    else:
        prop = punpy.MCPropagation(MCsteps, parallel_cores=parallel_cores)
        intp = Interpolator(
            method=method,
            method_hr=method_hr,
            relative=relative,
            min_scale=min_scale,
            extrapolate=extrapolate,
            add_model_error=include_model_uncertainties,
        )
        u_y_out, corr_y_out = prop.propagate_random(
            intp.interpolate_1d_along_example,
            [x_i, y_i, x_hr, y_hr, x],
            [None, u_y_i, None, u_y_hr, None],
            corr_x=[None, corr_y_i, None, corr_y_hr, None],
            return_corr=True,
        )

    if return_uncertainties:
        if return_corr:
            return y_out, u_y_out, corr_y_out
        else:
            return y_out, u_y_out

    else:
        return y_out


def interpolate_1d_along_example_nounc():
    pass