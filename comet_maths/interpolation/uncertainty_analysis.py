""" Module for functions to evaluate uncertainty associated with interpolation methods in interpolation module"""


from typing import List, Optional, Tuple, Union

import numpy as np

from comet_maths.interpolation import interpolate_1d

__author__ = ["Pieter De Vis <pieter.de.vis@npl.co.uk>"]
__all__ = ["default_unc_methods", "model_error_analytical_methods"]


def default_unc_methods(method: str) -> List[str]:
    """
    Function providing for each analytical interpolation method, the default methods that are compared to determine the model uncertainty for this interpolation method.

    :param method: method used in the interpolation
    :return: methods used to determine the model uncertainty on the provided method
    """
    if method.lower() in ["nearest", "previous", "next"]:
        unc_methods = ["nearest", "previous", "next", "linear"]
    elif method.lower() in ["linear", "quadratic", "cubic"]:
        unc_methods = ["linear", "quadratic", "cubic"]
    elif method.lower() in ["lagrange", "ius", "pchip"]:
        unc_methods = [method.lower(), "linear", "cubic"]
    else:
        raise ValueError(
            "comet_maths.interpolation: uncertainties for the model error for this interpolation method (%s) are not yet implemented"
            % (method)
        )
    return unc_methods


def model_error_analytical_methods(
    x_i: np.ndarray,
    y_i: np.ndarray,
    x: np.ndarray,
    unc_methods: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if unc_methods is None:  # use default unc methods if not specified
        unc_methods = ["linear", "quadratic", "cubic"]

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


if __name__ == "__main__":
    pass
