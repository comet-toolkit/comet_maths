""" Module for Gaussian process regression interpolation methods"""


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
import numpy as np

import comet_maths as cm


__author__ = ["Pieter De Vis <pieter.de.vis@npl.co.uk>"]
__all__ = ["gaussian_process_regression", "gpr_basics"]


def gaussian_process_regression(
    x_i,
    y_i,
    x,
    u_y_i=None,
    corr_y_i=None,
    kernel="RBF",
    min_scale=0.01,
    max_scale=10000,
    extrapolate="extrapolate",
    return_uncertainties=True,
    return_corr=False,
    include_model_uncertainties=True,
    add_model_error=False,
    MCsteps=100,
    parallel_cores=4,
):
    """
    Function to perform interpolation using Gaussian process regression

    :param x_i: Independent variable quantity x (coordinate data of y_i)
    :type x_i: np.ndarray
    :param y_i: measured variable quantity y (data to interpolate)
    :type y_i: np.ndarray
    :param x: Independent variable quantity x for which we are trying to obtain the measurand y
    :type x: np.ndarray
    :param u_y_i: uncertainties on y_i, defaults to None
    :type u_y_i: np.ndarray (optional)
    :param corr_y_i: error correlation matrix (can be "rand" for random, "syst" for systematic, or a custom 2D error correlation matrix), defaults to None
    :type corr_y_i: np.ndarray or str (optional)
    :param kernel: kernel to be used in the gpr interpolation. Defaults to "RBF".
    :type kernel: str (optional)
    :param min_scale: minimum bound on the scale parameter in the gaussian process regression. Defaults to 0.01
    :type min_scale: float (optional)
    :param max_scale: maximum bound on the scale parameter in the gaussian process regression. Defaults to 100
    :type max_scale: float (optional)
    :param extrapolate: extrapolation method, which can be set to "extrapolate" (in which case extrapolation is used using interpolation method defined in "method"), "nearest" (in which case nearest values are used for extrapolation), or "linear" (in which case linear extrapolation is used). Defaults to "extrapolate".
    :type extrapolate: str (optional)
    :param return_uncertainties: Boolean to indicate whether interpolation uncertainties should be calculated and returned. Defaults to False
    :type return_uncertainties: bool (optional)
    :param return_corr: Boolean to indicate whether interpolation error-correlation matrix should be calculated and returned. Defaults to False
    :type return_corr: bool (optional)
    :param include_model_uncertainties: Boolean to indicate whether model uncertainties should be added to output uncertainties to account for interpolation uncertainties. Not used for gpr. Defaults to True
    :type include_model_uncertainties: bool (optional)
    :param add_model_error: Boolean to indicate whether model error should be added to interpolated values to account for interpolation errors (useful in Monte Carlo approaches). Defaults to False
    :type add_model_error: bool (optional)
    :param MCsteps: number of MC iterations. Defaults to 100
    :type MCsteps: int (optional)
    :param parallel_cores: number of CPU to be used in parallel processing. Defaults to 4
    :type parallel_cores: int (optional)
    :return: The measurand y evaluated at the values x (interpolated data)
    :rtype: np.ndarray
    """
    # First calculate y_out without uncertainties
    y_out, cov_model = gpr_basics(
        x_i, y_i, x, kernel=kernel, min_scale=min_scale, max_scale=max_scale
    )

    y_out = redo_extrapolation(x_i, y_i, x, y_out, extrapolate)

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
            if (corr_y_i is None) or (corr_y_i == "rand"):
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
    x_i, y_i, x, u_y_i=None, kernel="RBF", min_scale=0.01, max_scale=10000,
):
    """
    Function to perform basic gaussian process regression

    :param x_i: Independent variable quantity x (coordinate data of y_i)
    :type x_i: np.ndarray
    :param y_i: measured variable quantity y (data to interpolate)
    :type y_i: np.ndarray
    :param x: Independent variable quantity x for which we are trying to obtain the measurand y
    :type x: np.ndarray
    :param u_y_i: uncertainties on y_i, defaults to None
    :type u_y_i: np.ndarray (optional)
    :param kernel: kernel to be used in the gpr interpolation. Defaults to "RBF".
    :type kernel: str (optional)
    :param min_scale: minimum bound on the scale parameter in the gaussian process regression. Defaults to 0.01
    :type min_scale: float (optional)
    :param max_scale: maximum bound on the scale parameter in the gaussian process regression. Defaults to 100
    :type max_scale: float (optional)
    :return: The measurand y evaluated at the values x (interpolated data)
    :rtype: np.ndarray
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


if __name__ == "__main__":
    pass
