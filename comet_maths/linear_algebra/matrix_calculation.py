"""Class with methods that convert a covariance or correlation matrix into a related metric"""

"""___Built-In Modules___"""
import comet_maths as cm

"""___Third-Party Modules___"""
import warnings
import numpy as np
import numdifftools as nd
from typing import List, Tuple, Union, Optional, Callable

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def calculate_Jacobian(
    fun: Callable,
    x: np.array,
    Jx_diag: Optional[bool] = False,
    step: Optional[float] = None,
) -> np.array:
    """
    Calculate the local Jacobian of function y=f(x) for a given value of x

    :param fun: flattened measurement function
    :param x: flattened local values of input quantities
    :param Jx_diag: Bool to indicate whether the Jacobian matrix can be described with semi-diagonal elements. With this we mean that the measurand has the same shape as each of the input quantities and the square jacobain between the measurand and each of the input quantities individually, only has diagonal elements. Defaults to False
    :param step: Defines the spacing used when calculating the Jacobian with numdifftools
    :return: Jacobian
    """
    Jfun = nd.Jacobian(fun, step=step)

    if Jx_diag:
        y = fun(x)
        Jfun = nd.Jacobian(fun)
        Jx = np.zeros((len(x), len(y)))
        # print(Jx.shape)
        for j in range(len(y)):
            xj = np.zeros(int(len(x) / len(y)))
            for i in range(len(xj)):
                xj[i] = x[i * len(y) + j]
            # print(xj.shape, xj)
            Jxj = Jfun(xj)
            for i in range(len(xj)):
                Jx[i * len(y) + j, j] = Jxj[0][i]
    else:
        Jx = Jfun(x)

    if len(Jx) != len(fun(x).flatten()):
        warnings.warn(
            "Dimensions of the Jacobian were flipped because its shape "
            "didn't match the shape of the output of the function "
            "(probably because there was only 1 input qty)."
        )
        Jx = Jx.T

    return Jx


def calculate_corr(
    MC_y: np.ndarray,
    corr_dims: Optional[int] = -99,
    PD_corr: Optional[bool] = True,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Calculate the correlation matrix between the MC-generated samples of output quantities.
    If corr_dims is specified, this axis will be the one used to calculate the correlation matrix (e.g. if corr_dims=0 and x.shape[0]=n, the correlation matrix will have shape (n,n)).
    This will be done for each combination of parameters in the other dimensions and the resulting correlation matrices are averaged.

    :param MC_y: MC-generated samples of the output quantities (measurands)
    :type MC_y: array
    :param corr_dims: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated. When the combined correlation of 2 or more (but not all) dimensions is required, they can be provided as a string containing the different dimension integers, separated by a dot (e.g. "0.2"). When multiple error_correlations should be calculated, they can be provided as a list.
    :type corr_dims: integer, optional
    :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True
    :type PD_corr: bool, optional
    :param dtype: numpy dtype for output variables
    :type dtype: numpy dtype
    :return: correlation matrix
    :rtype: array
    """
    if not hasattr(corr_dims, "__len__") or isinstance(corr_dims, str):
        corr_dims = [corr_dims]

    if len(MC_y.shape) < 3:
        corr_y = np.corrcoef(MC_y, rowvar=False)

    else:
        corr_y = np.empty(len(corr_dims), dtype=object)
        for i in range(len(corr_dims)):
            if corr_dims[i] is None:
                corr_y[i] = None
            elif isinstance(corr_dims[i], str):
                comb_axes = corr_dims[i].split(".")
                sli = [0] * MC_y.ndim
                sli[0] = slice(None)

                slib = [-1] * MC_y.ndim
                slib[0] = slice(None)

                for ii in range(len(comb_axes)):
                    sli[int(comb_axes[ii]) + 1] = slice(None)
                    slib[int(comb_axes[ii]) + 1] = slice(None)

                sli = tuple(sli)
                slib = tuple(slib)

                if len(corr_dims) == 1:
                    corr_y = np.corrcoef(
                        MC_y[sli].reshape((len(MC_y), -1)), rowvar=False, dtype=dtype
                    )
                    corr_yb = np.corrcoef(
                        MC_y[slib].reshape((len(MC_y), -1)), rowvar=False, dtype=dtype
                    )
                    if np.any((corr_y - corr_yb) > 0.05):
                        warnings.warn(
                            "comet_maths.matrix_calculation: The correlation matrix along the dimension with index %s is not constant (at least one element varies by more than 0.05 between first and last index of other dimensions). Are you sure it makes sense to use this dimension as a separate correlation dimension?"
                            % corr_dims
                        )

                else:
                    corr_y[i] = np.corrcoef(
                        MC_y[sli].reshape((len(MC_y), -1)), rowvar=False, dtype=dtype
                    )
                    corr_yb = np.corrcoef(
                        MC_y[slib].reshape((len(MC_y), -1)), rowvar=False, dtype=dtype
                    )
                    if np.any((corr_y[i] - corr_yb) > 0.05):
                        warnings.warn(
                            "comet_maths.matrix_calculation: The correlation matrix along the dimension with index %s is not constant (at least one element varies by more than 0.05 between first and last index of other dimensions). Are you sure it makes sense to use this dimension as a separate correlation dimension?"
                            % corr_dims
                        )

            elif corr_dims[i] >= 0:
                sli = [0] * MC_y.ndim
                sli[0] = slice(None)
                sli[corr_dims[i] + 1] = slice(None)

                slib = [-1] * MC_y.ndim
                slib[0] = slice(None)
                slib[corr_dims[i] + 1] = slice(None)

                sli = tuple(sli)
                slib = tuple(slib)

                if len(corr_dims) == 1:
                    corr_y = np.corrcoef(MC_y[sli], rowvar=False, dtype=dtype)
                    corr_yb = np.corrcoef(MC_y[slib], rowvar=False, dtype=dtype)
                    if np.any((corr_y - corr_yb) > 0.05):
                        warnings.warn(
                            "comet_maths.matrix_calculation: The correlation matrix along the dimension with index %s is not constant (at least one element varies by more than 0.05 between first and last index of other dimensions). Are you sure it makes sense to use this dimension as a separate correlation dimension?"
                            % corr_dims
                        )
                else:
                    corr_y[i] = np.corrcoef(MC_y[sli], rowvar=False, dtype=dtype)
                    corr_yb = np.corrcoef(MC_y[slib], rowvar=False, dtype=dtype)
                    if np.any((corr_y[i] - corr_yb) > 0.05):
                        warnings.warn(
                            "comet_maths.matrix_calculation: The correlation matrix along the dimension with index %s is not constant (at least one element varies by more than 0.05 between first and last index of other dimensions). Are you sure it makes sense to use this dimension as a separate correlation dimension?"
                            % corr_dims
                        )

            else:
                corr_y = np.corrcoef(
                    MC_y.reshape((len(MC_y), -1)), rowvar=False, dtype=dtype
                )

    if PD_corr and corr_y.ndim == 2:
        if not cm.isPD(corr_y):
            corr_y = cm.nearestPD_cholesky(corr_y, corr=True, return_cholesky=False)
    elif PD_corr and corr_y.ndim == 3:
        for i in range(len(corr_y)):
            if not cm.isPD(corr_y[i]):
                corr_y[i] = cm.nearestPD_cholesky(
                    corr_y[i], corr=True, return_cholesky=False
                )

    return corr_y


def nearestPD_cholesky(
    A: np.ndarray,
    diff: Optional[float] = 0.05,
    corr: Optional[bool] = False,
    return_cholesky: Optional[bool] = True,
) -> np.ndarray:
    """
    Find the nearest positive-definite matrix

    :param A: correlation matrix or covariance matrix
    :param diff: maximum difference that the error correlation matrix is allowed to be changed by to make it positive definite. Defaults to 0.001
    :param corr: boolean to indicate whether error correlation matrix is used (True) or error covariance matrix is used (False). Defaults to False
    :param return_cholesky: boolean to indicate whether the cholesky decomposition should be returned (True) or just the nearest positive definitive error correlation/covariance matrix (False). Defaults to True.
    :return: nearest positive-definite matrix

    Copied and adapted from [1] under BSD license.
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [2], which
    credits [3].
    [1] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    [2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [3] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    if isinstance(A, float) or np.count_nonzero(A) == 0:
        return A
    elif A.size == 1:
        return A

    B = (A + A.T) / 2
    try:
        _, s, V = np.linalg.svd(B)
    except:
        try:
            _, s, V = np.linalg.svd(B + 1.0e-6 * np.eye(A.shape[0]))
        except:
            warnings.warn("svd failed")
            V = None

    if V is None:
        A3 = B
    else:
        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

    try:
        chol = np.linalg.cholesky(A3)
        if return_cholesky:
            return chol
        else:
            return A3
    except:
        spacing = np.spacing(np.linalg.norm(A))
        if np.isnan(spacing):
            spacing = 0.001
        I = np.eye(A.shape[0])
        k = 1
        while not isPD(A3):
            try:
                mineig = np.min(np.real(np.linalg.eigvals(A3)))
                A3 += I * (-mineig * k**1.5 + spacing) / 2.0
                k += 1
            except:
                print(
                    A3,
                    np.count_nonzero(~np.isfinite(A3)),
                    np.count_nonzero(~np.isfinite(A)),
                    k,
                    spacing,
                )
                raise ValueError(
                    "Comet_maths was unable to make the error correlation positive definite (there might be nans due to zero uncertainty)"
                )
        if corr == True:
            A3 = cm.correlation_from_covariance(A3)
            maxdiff = np.max(np.abs(A - A3))
            if maxdiff > diff:
                raise ValueError(
                    "One of the correlation matrices is not postive definite (max diff=%s)."
                    "Correlation matrices need to be at least positive "
                    "semi-definite." % maxdiff
                )
            else:
                warnings.warn(
                    "One of the correlation matrices is not positive "
                    "definite. It has been slightly changed (maximum difference "
                    "of %s) to accomodate our method." % (maxdiff)
                )
                if return_cholesky:
                    return np.linalg.cholesky(A3)
                else:
                    return A3
        else:
            maxdiff = np.max(np.abs(A - A3) / (A3 + diff))
            if maxdiff > diff:
                raise ValueError(
                    "One of the correlation matrices is not postive definite (max diff=%s)."
                    "Correlation matrices need to be at least positive "
                    "semi-definite." % maxdiff
                )
            else:
                warnings.warn(
                    "One of the provided covariance matrix is not positive"
                    "definite. It has been slightly changed (maximum difference of "
                    "%s percent) to accomodate our method." % (maxdiff * 100)
                )
                if return_cholesky:
                    return np.linalg.cholesky(A3)
                else:
                    return A3


def isPD(B: np.ndarray) -> bool:
    """
    Returns true when input is positive-definite, via Cholesky

    :param B: matrix
    :return: true when input is positive-definite
    """
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
