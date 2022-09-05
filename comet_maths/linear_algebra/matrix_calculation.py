"""Class with methods that convert a covariance or correlation matrix into a related metric"""

"""___Built-In Modules___"""
import comet_maths as cm

"""___Third-Party Modules___"""
import warnings
import numpy as np
import numdifftools as nd

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def calculate_Jacobian(fun, x, Jx_diag=False, step=None):
    """
    Calculate the local Jacobian of function y=f(x) for a given value of x

    :param fun: flattened measurement function
    :type fun: function
    :param x: flattened local values of input quantities
    :type x: array
    :param Jx_diag: Bool to indicate whether the Jacobian matrix can be described with semi-diagonal elements. With this we mean that the measurand has the same shape as each of the input quantities and the square jacobain between the measurand and each of the input quantities individually, only has diagonal elements. Defaults to False
    :rtype Jx_diag: bool, optional
    :return: Jacobian
    :rtype: array
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


def calculate_corr(MC_y, corr_axis=-99, dtype=None):
    """
    Calculate the correlation matrix between the MC-generated samples of output quantities.
    If corr_axis is specified, this axis will be the one used to calculate the correlation matrix (e.g. if corr_axis=0 and x.shape[0]=n, the correlation matrix will have shape (n,n)).
    This will be done for each combination of parameters in the other dimensions and the resulting correlation matrices are averaged.

    :param MC_y: MC-generated samples of the output quantities (measurands)
    :type MC_y: array
    :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
    :type corr_axis: integer, optional
    :return: correlation matrix
    :rtype: array
    """
    # print("the shape is:",MC_y.shape)
    MCsteps = MC_y.shape[-1]

    if len(MC_y.shape) < 3:
        corr_y = np.corrcoef(MC_y)

    elif len(MC_y.shape) == 3:
        if corr_axis == 0:
            corr_ys = np.zeros((len(MC_y[:, 0, 0]), len(MC_y[:, 0, 0])), dtype=dtype)
            for i in range(len(MC_y[0])):
                corr_ys += np.corrcoef(MC_y[:, i])
            corr_y = corr_ys / len(MC_y[0])

        elif corr_axis == 1:
            corr_ys = np.zeros((len(MC_y[0, :, 0]), len(MC_y[0, :, 0])), dtype=dtype)
            # corr_ys = np.zeros(MC_y[0].shape)
            for i in range(len(MC_y)):
                corr_ys += np.corrcoef(MC_y[i])
            corr_y = corr_ys / len(MC_y)

        else:
            MC_y = MC_y.reshape((MC_y.shape[0] * MC_y.shape[1], MCsteps))
            corr_y = np.corrcoef(MC_y)

    elif len(MC_y.shape) == 4:
        if corr_axis == 0:
            corr_ys = np.zeros(
                (len(MC_y[:, 0, 0, 0]), len(MC_y[:, 0, 0, 0])), dtype=dtype
            )
            # corr_ys = np.zeros(MC_y[:, 0, 0].shape)
            for i in range(len(MC_y[0])):
                for j in range(len(MC_y[0, 0])):
                    corr_ys += np.corrcoef(MC_y[:, i, j])
            corr_y = corr_ys / (len(MC_y[0]) * len(MC_y[0, 0]))

        elif corr_axis == 1:
            corr_ys = np.zeros(
                (len(MC_y[0, :, 0, 0]), len(MC_y[0, :, 0, 0])), dtype=dtype
            )
            # corr_ys = np.zeros(MC_y[0, :, 0].shape)
            for i in range(len(MC_y)):
                for j in range(len(MC_y[0, 0])):
                    corr_ys += np.corrcoef(MC_y[i, :, j])
            corr_y = corr_ys / (len(MC_y) * len(MC_y[0, 0]))

        elif corr_axis == 2:
            corr_ys = np.zeros(
                (len(MC_y[0, 0, :, 0]), len(MC_y[0, 0, :, 0])), dtype=dtype
            )
            # corr_ys = np.zeros(MC_y[0, 0].shape)
            for i in range(len(MC_y)):
                for j in range(len(MC_y[0])):
                    corr_ys += np.corrcoef(MC_y[i, j])
            corr_y = corr_ys / (len(MC_y) * len(MC_y[0]))
        else:
            MC_y = MC_y.reshape(
                (MC_y.shape[0] * MC_y.shape[1] * MC_y.shape[2], MCsteps)
            )
            corr_y = np.corrcoef(MC_y)
    elif len(MC_y.shape) == 5:
        if corr_axis == 0:
            corr_ys = np.zeros(
                (len(MC_y[:, 0, 0, 0, 0]), len(MC_y[:, 0, 0, 0, 0])), dtype=dtype
            )
            # corr_ys = np.zeros(MC_y[:, 0, 0].shape)
            for i in range(len(MC_y[0])):
                for j in range(len(MC_y[0, 0])):
                    corr_ys += np.corrcoef(MC_y[:, i, j])
            corr_y = corr_ys / (len(MC_y[0]) * len(MC_y[0, 0]))

        elif corr_axis == 1:
            corr_ys = np.zeros(
                (len(MC_y[0, :, 0, 0, 0]), len(MC_y[0, :, 0, 0, 0])), dtype=dtype
            )
            # corr_ys = np.zeros(MC_y[0, :, 0].shape)
            for i in range(len(MC_y)):
                for j in range(len(MC_y[0, 0])):
                    corr_ys += np.corrcoef(MC_y[i, :, j])
            corr_y = corr_ys / (len(MC_y) * len(MC_y[0, 0]))

        elif corr_axis == 2:
            corr_ys = np.zeros(
                (len(MC_y[0, 0, :, 0, 0]), len(MC_y[0, 0, :, 0, 0])), dtype=dtype
            )
            # corr_ys = np.zeros(MC_y[0, 0].shape)
            for i in range(len(MC_y)):
                for j in range(len(MC_y[0])):
                    corr_ys += np.corrcoef(MC_y[i, j])
            corr_y = corr_ys / (len(MC_y) * len(MC_y[0]))

        elif corr_axis == 3:
            sli = tuple(
                [slice(None) if (idim == corr_axis) else 0 for idim in range(MC_y.ndim)]
            )

            corr_ys = np.zeros((len(MC_y[sli]), len(MC_y[sli])), dtype=dtype)
            # corr_ys = np.zeros(MC_y[0, 0].shape)
            for i in range(len(MC_y)):
                for j in range(len(MC_y[0])):
                    corr_ys += np.corrcoef(MC_y[i, j])
            corr_y = corr_ys / (len(MC_y) * len(MC_y[0]))
        else:
            MC_y = MC_y.reshape(
                (MC_y.shape[0] * MC_y.shape[1] * MC_y.shape[2] * MC_y.shape[3], MCsteps)
            )
            corr_y = np.corrcoef(MC_y)
    else:
        raise ValueError(
            "punpy.mc_propagation: MC_y has too high dimensions. Reduce the dimensionality of the input data"
        )

    return corr_y


def nearestPD_cholesky(A, diff=0.001, corr=False, return_cholesky=True):
    """
    Find the nearest positive-definite matrix

    :param A: correlation matrix or covariance matrix
    :type A: array
    :return: nearest positive-definite matrix
    :rtype: array

    Copied and adapted from [1] under BSD license.
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [2], which
    credits [3].
    [1] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    [2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [3] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    try:
        _, s, V = np.linalg.svd(B)
    except:
        _, s, V = np.linalg.svd(B + 1.0e-6 * np.eye(A.shape[0]))

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

        I = np.eye(A.shape[0])
        k = 1
        while not isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        if corr == True:
            A3 = cm.correlation_from_covariance(A3)
            maxdiff = np.max(np.abs(A - A3))
            if maxdiff > diff:
                raise ValueError(
                    "One of the correlation matrices is not postive definite. "
                    "Correlation matrices need to be at least positive "
                    "semi-definite."
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
                    "One of the provided covariance matrices is not postive "
                    "definite. Covariance matrices need to be at least positive "
                    "semi-definite. Please check your covariance matrix."
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


def isPD(B):
    """
    Returns true when input is positive-definite, via Cholesky

    :param B: matrix
    :type B: array
    :return: true when input is positive-definite
    :rtype: bool
    """
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
