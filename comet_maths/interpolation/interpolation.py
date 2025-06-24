""" Module for interpolation of data and enables propagation of uncertainties through the interpolation."""

from typing import Union, Optional, List, Tuple

from scipy.interpolate import InterpolatedUnivariateSpline, PchipInterpolator
from scipy.interpolate import interp1d
from scipy.interpolate import lagrange

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C

import matplotlib.pyplot as plt
import numpy as np
import punpy

import comet_maths as cm

__author__ = ["Pieter De Vis <pieter.de.vis@npl.co.uk>"]
__all__ = ["Interpolator", "interpolate_1d", "interpolate_1d_along_example"]


class Interpolator:
    """
    Class to provide a set of interpolation methods for the interpolation of data.
    The class provides a range of interpolation methods, and enables propagation of
    uncertainties through the interpolation by providing measurement functions that
    only take the numerical input quantities as arguments. All the other options for
    interpolation are stored in the class attributes.

    :param relative: Boolean to indicate whether a relative normalisation (True) or absolute normalisation (False) should be used. Defaults to True.
    :param method: Sting to indicate which interpolation method should be used to interpolate between normalised data (core interpolation step within the approach). Defaults to Gaussian Progress Regression.
    :param method_hr: String to indicate which interpolation method should be used to interpolate between high resolution measurements. Defaults to cubic spline interpolation.
    :param unc_methods: interpolation methods to use in the calculation of the model error for interpolation between normalised data. Not used for gpr. Defaults to None, in which case a standard list is used for each interpolation method.
    :param unc_methods_hr: interpolation methods to use in the calculation of the model error for interpolation between high resolution measurements. Not used for gpr. Defaults to None, in which case a standard list is used for each interpolation method.
    :param min_scale: minimum bound on the scale parameter in the gaussian process regression. Only used if gpr is selected as method. Defaults to 0.3
    :param extrapolate: extrapolation method, which can be set to "extrapolate" (in which case extrapolation is used using interpolation method defined in "method"), "nearest" (in which case nearest values are used for extrapolation), or "linear" (in which case linear extrapolation is used). Defaults to "extrapolate".
    :param add_model_error: Boolean to indicate whether model error should be added to interpolated values to account for interpolation errors (useful in Monte Carlo approaches). Defaults to False
    :param plot_residuals: Boolean to indicate whether a plot of the residuals should be made (and stored as residuals.png). Defaults to False
    """

    def __init__(
        self,
        relative: Optional[bool] = True,
        method: Optional[str] = "cubic",
        method_hr: Optional[str] = "cubic",
        unc_methods: Optional[List[str]] = None,
        unc_methods_hr: Optional[List[str]] = None,
        min_scale: Optional[float] = 0.3,
        extrapolate: Optional[str] = "nearest",
        add_model_error: Optional[bool] = True,
        plot_residuals: Optional[bool] = False,
    ) -> None:
        """ """
        self.relative = relative
        self.method = method
        self.method_hr = method_hr
        self.add_model_error = add_model_error
        self.min_scale = min_scale
        self.extrapolate = extrapolate
        self.plot_residuals = plot_residuals
        self.unc_methods = unc_methods
        self.unc_methods_hr = unc_methods_hr

    def interpolate_1d_along_example(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        x_hr: np.ndarray,
        y_hr: np.ndarray,
        x: np.ndarray,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Method for interpolating between datapoints by following an example.
        The example can come from either models or higher-resolution observations.
        Here the example is assumed to have an unknown normalisation or poor absolute calibration,
        yet the low resolution data has a more precise calibration (and can thus be used to constrain the high-resolution model).

        :param x_i: Independent variable quantity x for the low resolution data
        :param y_i: measured variable quantity y for the low resolution data
        :param x_hr: Independent variable quantity x for the high resolution data
        :param y_hr: measured variable quantity y for the high resolution data
        :param x: Independent variable quantity x for which we are trying to obtain the measurand y
        :return: The measurand y evaluated at the values x
        """
        return interpolate_1d_along_example(
            x_i,
            y_i,
            x_hr,
            y_hr,
            x,
            relative=self.relative,
            method=self.method,
            method_hr=self.method_hr,
            unc_methods=self.unc_methods,
            unc_methods_hr=self.unc_methods_hr,
            u_y_i=None,
            corr_y_i=None,
            u_y_hr=None,
            corr_y_hr=None,
            min_scale=self.min_scale,
            extrapolate=self.extrapolate,
            return_uncertainties=False,
            return_corr=False,
            add_model_error=self.add_model_error,
            plot_residuals=self.plot_residuals,
        )

    def interpolate_1d(
        self, x_i: np.ndarray, y_i: np.ndarray, x: np.ndarray
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Interpolates 1D data to defined coordinates x in 1D

        :param x_i: Independent variable quantity x (coordinate data of y_i)
        :param y_i: measured variable quantity y (data to interpolate)
        :param x: Independent variable quantity x for which we are trying to obtain the measurand y
        :return: The measurand y_i evaluated at the values x
        """

        return interpolate_1d(
            x_i,
            y_i,
            x,
            method=self.method,
            unc_methods=self.unc_methods,
            return_uncertainties=False,
            return_corr=False,
            min_scale=self.min_scale,
            extrapolate=self.extrapolate,
            add_model_error=self.add_model_error,
        )


# def interpolate(
#     x_i, y_i, x, method="linear", return_uncertainties=False, add_model_error=False, include_model_uncertainties=True,
# ):
#     """
#     Interpolates data to defined coordinates x
#
#     :param x_i: initial coordinate data of y_i
#     :type x_i: np.ndarray
#     :param y_i: data to interpolate
#     :type y_i: np.ndarray
#     :param x: coordinate data to interpolate y_i to
#     :type x: np.ndarray
#     :param method: interpolation method to be used, defaults to linear
#     :type method: string (optional)
#     :param return_uncertainties: Boolean to indicate whether interpolation (model error) uncertainties should be calculated and returned. Defaults to False
#     :type return_uncertainties: bool (optional)
#     :param add_model_error: Boolean to indicate whether model error should be added to account for interpolation uncertainties (useful in Monte Carlo approaches). Defaults to False
#     :type add_model_error: bool (optional)
#     :return: interpolate data
#     :rtype: np.ndarray
#     """
#     x_i = np.array(x_i)
#     y_i = np.array(y_i)
#
#     y_i = y_i[~np.isnan(x_i)]
#     x_i = x_i[~np.isnan(x_i)]
#
#     x_i = x_i[~np.isnan(y_i)]
#     y_i = y_i[~np.isnan(y_i)]
#
#     if x_i.ndim == 1:
#         interpolate_1d(x_i, y_i, x, method, return_uncertainties=return_uncertainties, add_model_error=add_model_error, include_model_uncertainties=include_model_uncertainties)
#
#     else:
#         raise NotImplementedError()


def interpolate_1d(
    x_i: np.ndarray,
    y_i: np.ndarray,
    x: np.ndarray,
    method: Optional[str] = "linear",
    unc_methods: Optional[List[str]] = None,
    u_y_i: Optional[np.ndarray] = None,
    corr_y_i: Optional[Union[np.ndarray, str]] = None,
    min_scale: Optional[float] = 0.3,
    extrapolate: Optional[str] = "extrapolate",
    return_uncertainties: Optional[bool] = False,
    return_corr: Optional[bool] = False,
    include_model_uncertainties: Optional[bool] = True,
    add_model_error: Optional[bool] = False,
    MCsteps: Optional[int] = 50,
    parallel_cores: Optional[int] = 4,
    interpolate_axis: Optional[int] = 0,
) -> Union[
    np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Interpolates 1D data to defined coordinates x in 1D

    :param x_i: Independent variable quantity x (coordinate data of y_i)
    :param y_i: measured variable quantity y (data to interpolate)
    :param x: Independent variable quantity x for which we are trying to obtain the measurand y
    :param method: interpolation method to be used, defaults to linear
    :param unc_methods: interpolation methods to use in the calculation of the model error. Not used for gpr. Defaults to None, in which case a standard list is used for each interpolation method.
    :param u_y_i: uncertainties on y_i, defaults to None
    :param corr_y_i: error correlation matrix (can be "rand" for random, "syst" for systematic, or a custom 2D error correlation matrix), defaults to None
    :param min_scale: minimum bound on the scale parameter in the gaussian process regression. Only used if gpr is selected as method. Defaults to 0.3
    :param extrapolate: extrapolation method, which can be set to "extrapolate" (in which case extrapolation is used using interpolation method defined in "method"), "nearest" (in which case nearest values are used for extrapolation), or "linear" (in which case linear extrapolation is used). Defaults to "extrapolate".
    :param return_uncertainties: Boolean to indicate whether interpolation uncertainties should be calculated and returned. Defaults to False
    :param return_corr: Boolean to indicate whether interpolation error-correlation matrix should be calculated and returned. Defaults to False
    :param include_model_uncertainties: Boolean to indicate whether model uncertainties should be added to output uncertainties to account for interpolation uncertainties. Not used for gpr. Defaults to True
    :param add_model_error: Boolean to indicate whether model error should be added to interpolated values to account for interpolation errors (useful in Monte Carlo approaches). Defaults to False
    :param MCsteps: number of MC iterations. Defaults to 100
    :param parallel_cores: number of CPU to be used in parallel processing. Defaults to 4
    :return: The measurand y evaluated at the values x (interpolated data)
    """
    if method.lower() in ["gpr", "gaussian_process_regression"]:
        if x_i.shape != y_i.shape:
            raise NotImplementedError(
                "The provided x_i and y_i need to be 1 dimensional to use this method"
            )

        return gaussian_process_regression(
            x_i,
            y_i,
            x,
            min_scale=min_scale,
            u_y_i=u_y_i,
            corr_y_i=corr_y_i,
            return_uncertainties=return_uncertainties,
            add_model_error=add_model_error,
            return_corr=return_corr,
            extrapolate=extrapolate,
            MCsteps=MCsteps,
            parallel_cores=parallel_cores,
        )

    elif add_model_error:
        if unc_methods is None:
            unc_methods = default_unc_methods(method)

        extrapolate_methods = [extrapolate, "nearest"]
        return interpolate_1d(
            x_i,
            y_i,
            x,
            method=np.random.choice(unc_methods, 1)[0],
            extrapolate=np.random.choice(extrapolate_methods, 1)[0],
            return_uncertainties=False,
            add_model_error=False,
            interpolate_axis=interpolate_axis,
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
        f_i = interp1d(
            x_i,
            y_i,
            kind=method.lower(),
            fill_value="extrapolate",
            axis=interpolate_axis,
        )
        y = f_i(x).squeeze()

    elif method.lower() in ["ius", "lagrange", "pchip"]:
        if x_i.shape != y_i.shape:
            raise NotImplementedError(
                "The provided x_i and y_i need to be 1 dimensional to use this method"
            )
        if method.lower() == "ius":
            f_i = InterpolatedUnivariateSpline(x_i, y_i, ext=0)
            y = f_i(x).squeeze()

        elif method.lower() == "lagrange":
            f_i = lagrange(x_i, y_i)
            y = f_i(x).squeeze()

        elif method.lower() == "pchip":
            f_i = PchipInterpolator(x_i, y_i)
            y = f_i(x).squeeze()

    else:
        raise ValueError(
            "comet_maths.interpolation: this interpolation method (%s) is not implemented"
            % (method)
        )

    y = _redo_extrapolation(x_i, y_i, x, y, extrapolate)

    if (not return_corr) and (not return_uncertainties):
        return y

    else:
        if unc_methods is None:
            unc_methods = default_unc_methods(method)

        if include_model_uncertainties:
            u_y_model, corr_y_model, cov_model = model_error_analytical_methods(
                x_i, y_i, x, unc_methods=unc_methods
            )

        if (not return_uncertainties) and (not return_corr):
            return y

        if u_y_i is None:
            if include_model_uncertainties:
                u_y_model, corr_y_model, cov_model = model_error_analytical_methods(
                    x_i, y_i, x, unc_methods=unc_methods
                )
                y_unc = u_y_model
                y_corr = corr_y_model

            else:
                y_unc = None
                y_corr = None

        else:
            prop = punpy.MCPropagation(MCsteps, parallel_cores=1)
            intp = Interpolator(
                unc_methods=unc_methods, add_model_error=include_model_uncertainties
            )

            if return_corr:
                y_unc, y_corr = prop.propagate_random(
                    intp.interpolate_1d,
                    [x_i, y_i, x],
                    [None, u_y_i, None],
                    corr_x=[None, corr_y_i, None],
                    return_corr=return_corr,
                )
                y_corr[np.where(np.isnan(y_corr))] = 0

            else:
                y_unc = prop.propagate_random(
                    intp.interpolate_1d,
                    [x_i, y_i, x],
                    [None, u_y_i, None],
                    corr_x=[None, corr_y_i, None],
                    return_corr=return_corr,
                )

        if return_uncertainties and return_corr:
            return y, y_unc, y_corr
        else:
            return y, y_unc


def default_unc_methods(method: str) -> List[str]:
    """
    Function providing for each analytical interpolation method, the default methods that are compared to determine the model uncertainty for this interpolation method.

    :param method: method used in the interpolation
    :return: methods used to determine the model uncertainty on the provided method
    """
    if method.lower() in ["nearest", "previous", "next"]:
        unc_methods = ["nearest", "previous", "next", "linear"]
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
    elif method.lower() == "pchip":
        unc_methods = ["pchip", "linear", "cubic"]
    else:
        raise ValueError(
            "comet_maths.interpolation: uncertainties for the model error for this interpolation method (%s) are not yet implemented"
            % (method)
        )
    return unc_methods


def _redo_extrapolation(
    x_i: np.ndarray,
    y_i: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    extrapolate: Optional[str] = "extrapolate",
) -> np.ndarray:
    """
    function to check if extrapolate is "nearest" or "linear, and if so, redo the extrapolation

    :param x_i: Independent variable quantity x (coordinate data of y_i)
    :param y_i: measured variable quantity y (data to interpolate)
    :param x: Independent variable quantity x for which we are trying to obtain the measurand y
    :param y: interpolated values using standard method
    :param extrapolate: extrapolation method, which can be set to "extrapolate" (in which case extrapolation is used using interpolation method defined in "method"), "nearest" (in which case nearest values are used for extrapolation), or "linear" (in which case linear extrapolation is used). Defaults to "extrapolate".
    :return: interpolated values with correct extrapolation
    """
    if hasattr(x,"__len__") and len(x) > 1:
        if extrapolate == "nearest":
            y[x < x_i[0]] = y_i[0]
            y[x > x_i[-1]] = y_i[-1]

        elif extrapolate == "linear":
            f_lin = interp1d(x_i, y_i, kind="linear", fill_value="extrapolate")
            y[x < x_i[0]] = f_lin(x[x < x_i[0]])
            y[x > x_i[-1]] = f_lin(x[x > x_i[-1]])
    else:
        if extrapolate == "nearest":
            if x < x_i[0]:
                y = y_i[0]
            elif x > x_i[-1]:
                y = y_i[-1]

        elif extrapolate == "linear":
            if x < x_i[0] or x > x_i[-1]:
                f_lin = interp1d(x_i, y_i, kind="linear", fill_value="extrapolate")
                y = f_lin(x)

    return y


def model_error_analytical_methods(
    x_i: np.ndarray,
    y_i: np.ndarray,
    x: np.ndarray,
    unc_methods: List[str] = ["linear", "quadratic", "cubic"],
) -> np.ndarray:
    """
    Function to calculate the interpolation model uncertainty by calculating the standard
    deviation between various interpolation methods. Also includes extrapolation
    uncertainty when appropriate.

    :param x_i: Independent variable quantity x (coordinate data of y_i)
    :param y_i: measured variable quantity y (data to interpolate)
    :param x: Independent variable quantity x for which we are trying to obtain the measurand y
    :param unc_methods: interpolation methods to use in the calculation of the model error. Not used for gpr. Defaults to None, in which case a standard list is used for each interpolation method.
    :return: interpolation model uncertainty
    """

    # check if any values need to be extrapolated, and if so include both extrapolation methods
    if (x[0] < x_i[0]) or (x[-1] > x_i[-1]):
        data = np.zeros((len(x), 2 * len(unc_methods)))
        for i in range(len(unc_methods)):
            data[:, i] = interpolate_1d(
                x_i, y_i, x, unc_methods[i], extrapolate="nearest"
            )
            data[:, i + len(unc_methods)] = interpolate_1d(
                x_i, y_i, x, unc_methods[i], extrapolate="extrapolate"
            )

    else:
        data = np.zeros((len(x), len(unc_methods)))
        for i in range(len(unc_methods)):
            data[:, i] = interpolate_1d(x_i, y_i, x, unc_methods[i])
    return np.std(data, axis=1), np.corrcoef(data), np.cov(data)


def gaussian_process_regression(
    x_i: np.ndarray,
    y_i: np.ndarray,
    x: np.ndarray,
    u_y_i: Optional[np.ndarray] = None,
    corr_y_i: Optional[Union[np.ndarray, str]] = None,
    kernel: Optional[str] = "RBF",
    min_scale: Optional[float] = 0.01,
    max_scale: Optional[float] = 10000,
    extrapolate: Optional[str] = "extrapolate",
    return_uncertainties: Optional[bool] = True,
    return_corr: Optional[bool] = False,
    include_model_uncertainties: Optional[bool] = True,
    add_model_error: Optional[bool] = False,
    MCsteps: Optional[int] = 100,
    parallel_cores: Optional[int] = 4,
) -> np.ndarray:
    """
    Function to perform interpolation using Gaussian process regression

    :param x_i: Independent variable quantity x (coordinate data of y_i)
    :param y_i: measured variable quantity y (data to interpolate)
    :param x: Independent variable quantity x for which we are trying to obtain the measurand y
    :param u_y_i: uncertainties on y_i, defaults to None
    :param corr_y_i: error correlation matrix (can be "rand" for random, "syst" for systematic, or a custom 2D error correlation matrix), defaults to None
    :param kernel: kernel to be used in the gpr interpolation. Defaults to "RBF".
    :param min_scale: minimum bound on the scale parameter in the gaussian process regression. Defaults to 0.01
    :param max_scale: maximum bound on the scale parameter in the gaussian process regression. Defaults to 100
    :param extrapolate: extrapolation method, which can be set to "extrapolate" (in which case extrapolation is used using interpolation method defined in "method"), "nearest" (in which case nearest values are used for extrapolation), or "linear" (in which case linear extrapolation is used). Defaults to "extrapolate".
    :param return_uncertainties: Boolean to indicate whether interpolation uncertainties should be calculated and returned. Defaults to False
    :param return_corr: Boolean to indicate whether interpolation error-correlation matrix should be calculated and returned. Defaults to False
    :param include_model_uncertainties: Boolean to indicate whether model uncertainties should be added to output uncertainties to account for interpolation uncertainties. Not used for gpr. Defaults to True
    :param add_model_error: Boolean to indicate whether model error should be added to interpolated values to account for interpolation errors (useful in Monte Carlo approaches). Defaults to False
    :param MCsteps: number of MC iterations. Defaults to 100
    :param parallel_cores: number of CPU to be used in parallel processing. Defaults to 4
    :return: The measurand y evaluated at the values x (interpolated data)
    """
    # First calculate y_out without uncertainties
    y_out, cov_model = gpr_basics(
        x_i, y_i, x, kernel=kernel, min_scale=min_scale, max_scale=max_scale
    )

    y_out = _redo_extrapolation(x_i, y_i, x, y_out, extrapolate)

    if add_model_error:
        y_out = cm.generate_sample_cov(1, y_out, cov_model, diff=0.1).squeeze()

    if (not return_uncertainties) and (not return_corr):
        return y_out

    if (u_y_i is None) and include_model_uncertainties:
        corr_y_out = cm.correlation_from_covariance(cov_model)
        u_y_out = cm.uncertainty_from_covariance(cov_model)

    else:
        # next determine if a simple uncertainty from gpr is possible or if MC is necessary
        uncertainties_simple = False
        if (u_y_i is not None) and include_model_uncertainties:
            if (corr_y_i is None) or (
                isinstance(corr_y_i, str) and (corr_y_i == "rand")
            ):
                uncertainties_simple = True
            elif isinstance(corr_y_i, np.ndarray):
                if np.all(corr_y_i == np.diag(corr_y_i)):
                    uncertainties_simple = True

        if uncertainties_simple:
            y_out_simple, cov_simple = gpr_basics(
                x_i,
                y_i,
                x,
                u_y_i=u_y_i,
                kernel=kernel,
                min_scale=min_scale,
                max_scale=max_scale,
            )
            corr_y_out = cm.correlation_from_covariance(cov_simple)
            u_y_out = cm.uncertainty_from_covariance(cov_simple)

        else:
            prop = punpy.MCPropagation(MCsteps, parallel_cores=parallel_cores)
            intp = Interpolator(
                method="gpr",
                min_scale=min_scale,
                add_model_error=include_model_uncertainties,
            )
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


def gpr_basics(
    x_i: np.ndarray,
    y_i: np.ndarray,
    x: np.ndarray,
    u_y_i: Optional[np.ndarray] = None,
    kernel: Optional[str] = "RBF",
    min_scale: Optional[float] = 0.01,
    max_scale: Optional[float] = 10000,
) -> np.ndarray:
    """
    Function to perform basic gaussian process regression

    :param x_i: Independent variable quantity x (coordinate data of y_i)
    :param y_i: measured variable quantity y (data to interpolate)
    :param x: Independent variable quantity x for which we are trying to obtain the measurand y
    :param u_y_i: uncertainties on y_i, defaults to None
    :param kernel: kernel to be used in the gpr interpolation. Defaults to "RBF".
    :param min_scale: minimum bound on the scale parameter in the gaussian process regression. Defaults to 0.01
    :param max_scale: maximum bound on the scale parameter in the gaussian process regression. Defaults to 100
    :return: The measurand y evaluated at the values x (interpolated data)
    """
    X = np.atleast_2d(x_i).T

    # Observations
    y = np.atleast_2d(y_i).T

    if u_y_i is None:
        alpha = 1e-10  # default value for GaussianProcessRegressor
    else:
        alpha = u_y_i**2

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
    x_i: np.ndarray,
    y_i: np.ndarray,
    x_hr: np.ndarray,
    y_hr: np.ndarray,
    x: np.ndarray,
    relative: Optional[bool] = True,
    method: Optional[str] = "linear",
    method_hr: Optional[str] = "linear",
    unc_methods: Optional[List[str]] = None,
    unc_methods_hr: Optional[List[str]] = None,
    u_y_i: Optional[np.ndarray] = None,
    corr_y_i: Optional[Union[str, np.ndarray]] = None,
    u_y_hr: Optional[np.ndarray] = None,
    corr_y_hr: Optional[Union[str, np.ndarray]] = None,
    min_scale: Optional[float] = 0.3,
    extrapolate: Optional[str] = "nearest",
    return_uncertainties: Optional[bool] = False,
    return_corr: Optional[bool] = False,
    include_model_uncertainties: Optional[bool] = True,
    add_model_error: Optional[bool] = False,
    plot_residuals: Optional[bool] = False,
    MCsteps: Optional[int] = 50,
    parallel_cores: Optional[int] = 4,
) -> Union[
    np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Method for interpolating between datapoints by following an example.
    The example can come from either models or higher-resolution observations.
    Here the example is assumed to have an unknown normalisation or poor absolute calibration,
    yet the low resolution data has a more precise calibration (and can thus be used to constrain the high-resolution model).

    :param x_i: Independent variable quantity x for the low resolution data
    :param y_i: measured variable quantity y for the low resolution data
    :param x_hr: Independent variable quantity x for the high resolution data
    :param y_hr: measured variable quantity y for the high resolution data
    :param x: Independent variable quantity x for which we are trying to obtain the measurand y
    :param relative: Boolean to indicate whether a relative normalisation (True) or absolute normalisation (False) should be used. Defaults to True.
    :param method: Sting to indicate which interpolation method should be used to interpolate between normalised data (core interpolation step within the approach). Defaults to Gaussian Progress Regression.
    :param method_hr: String to indicate which interpolation method should be used to interpolate between high resolution measurements. Defaults to cubic spline interpolation.
    :param unc_methods: interpolation methods to use in the calculation of the model error for interpolation between normalised data. Not used for gpr. Defaults to None, in which case a standard list is used for each interpolation method.
    :param unc_methods_hr: interpolation methods to use in the calculation of the model error for interpolation between high resolution measurements. Not used for gpr. Defaults to None, in which case a standard list is used for each interpolation method.
    :param u_y_i: uncertainties on y_i, defaults to None
    :param corr_y_i: error correlation matrix for u_y_i (can be "rand" for random, "syst" for systematic, or a custom 2D error correlation matrix), defaults to None
    :param u_y_hr: uncertainties on y_hr, defaults to None
    :param corr_y_hr: error correlation matrix for u_y_hr (can be "rand" for random, "syst" for systematic, or a custom 2D error correlation matrix), defaults to None
    :param min_scale: minimum bound on the scale parameter in the gaussian process regression. Only used if gpr is selected as method. Defaults to 0.3
    :param extrapolate: extrapolation method, which can be set to "extrapolate" (in which case extrapolation is used using interpolation method defined in "method"), "nearest" (in which case nearest values are used for extrapolation), or "linear" (in which case linear extrapolation is used). Defaults to "extrapolate".
    :param return_uncertainties: Boolean to indicate whether interpolation uncertainties should be calculated and returned. Defaults to False
    :param return_corr: Boolean to indicate whether interpolation error-correlation matrix should be calculated and returned. Defaults to False
    :param include_model_uncertainties: Boolean to indicate whether model uncertainties should be added to output uncertainties to account for interpolation uncertainties. Not used for gpr. Defaults to True
    :param add_model_error: Boolean to indicate whether model error should be added to interpolated values to account for interpolation errors (useful in Monte Carlo approaches). Defaults to False
    :param plot_residuals: Boolean to indicate whether a plot of the residuals should be made (and stored as residuals.png). Defaults to False
    :param MCsteps: number of MC iterations. Defaults to 100
    :param parallel_cores: number of CPU to be used in parallel processing. Defaults to 4
    :return: The measurand y evaluated at the values x (interpolated values), optionally with correlation and uncertainties if specified
    """
    y_hr_i = interpolate_1d(
        x_hr,
        y_hr,
        x_i,
        method=method_hr,
        unc_methods=unc_methods_hr,
        min_scale=min_scale,
        extrapolate=extrapolate,
        add_model_error=add_model_error,
    )

    if x.shape == x_hr.shape and np.all(x == x_hr):
        y_hr_out = y_hr
    else:
        y_hr_out = interpolate_1d(
            x_hr,
            y_hr,
            x,
            method=method_hr,
            unc_methods=unc_methods_hr,
            min_scale=min_scale,
            extrapolate=extrapolate,
            add_model_error=add_model_error,
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
        unc_methods=unc_methods,
        min_scale=min_scale,
        extrapolate=extrapolate,
        add_model_error=add_model_error,
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

    else:
        prop = punpy.MCPropagation(MCsteps, parallel_cores=parallel_cores)
        intp = Interpolator(
            relative=relative,
            method=method,
            method_hr=method_hr,
            unc_methods=unc_methods,
            unc_methods_hr=unc_methods_hr,
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


if __name__ == "__main__":
    pass
