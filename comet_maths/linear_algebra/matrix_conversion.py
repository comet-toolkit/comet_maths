"""Class with methods that convert a covariance or correlation matrix into a related metric"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import warnings
import numpy as np

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def correlation_from_covariance(covariance):
    """
    Convert covariance matrix to correlation matrix

    :param covariance: Covariance matrix
    :type covariance: array
    :return: Correlation matrix
    :rtype: array
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def uncertainty_from_covariance(covariance):
    """
    Convert covariance matrix to uncertainty

    :param covariance: Covariance matrix
    :type covariance: array
    :return: uncertainties
    :rtype: array
    """
    return np.sqrt(np.diag(covariance))


def convert_corr_to_cov(corr, u):
    """
    Convert correlation matrix to covariance matrix

    :param corr: correlation matrix
    :type corr: array
    :param u: uncertainties
    :type u: array
    :return: covariance matrix
    :rtype: array
    """
    return u.reshape((-1, 1)) * corr * (u.reshape((1, -1)))


def convert_cov_to_corr(cov, u):
    """
    Convert covariance matrix to correlation matrix

    :param corr: covariance matrix
    :type corr: array
    :param u: uncertainties
    :type u: array
    :return: correlation matrix
    :rtype: array
    """
    return 1 / u.reshape((-1, 1)) * cov / (u.reshape((1, -1)))


def calculate_flattened_corr(corrs, corr_between):
    """
    Combine correlation matrices for different input quantities, with a correlation
    matrix that gives the correlation between the input quantities into a full
    (flattened) correlation matrix combining the two.

    :param corrs: list of correlation matrices for each input quantity
    :type corrs: list[array]
    :param corr_between: correlation matrix between the input quantities
    :type corr_between: array
    :return: full correlation matrix combining the correlation matrices
    :rtype: array
    """
    totcorrlen = 0
    for i in range(len(corrs)):
        totcorrlen += len(corrs[i])
    totcorr = np.eye(totcorrlen)
    for i in range(len(corrs)):
        for j in range(len(corrs)):
            if corr_between[i, j] > 0:
                ist = i * len(corrs[i])
                iend = (i + 1) * len(corrs[i])
                jst = j * len(corrs[j])
                jend = (j + 1) * len(corrs[j])
                totcorr[ist:iend, jst:jend] = (
                    corr_between[i, j] * corrs[i] ** 0.5 * corrs[j] ** 0.5
                )
    return totcorr


def separate_flattened_corr(corr, ndim):
    """
    Separate a full (flattened) correlation matrix into a list of correlation matrices
    for each output variable and a correlation matrix between the output variables.

    :param corr: full correlation matrix
    :type corr: array
    :param ndim: number of output variables
    :type ndim: int
    :return: list of correlation matrices for each output variable, correlation matrix between the output variables
    :type corrs: list[array]
    :rtype: list[array], array
    """

    corrs = np.empty(ndim, dtype=object)
    for i in range(ndim):
        corrs[i] = correlation_from_covariance(
            corr[
                int(i * len(corr) / ndim) : int((i + 1) * len(corr) / ndim),
                int(i * len(corr) / ndim) : int((i + 1) * len(corr) / ndim),
            ]
        )

    corrs_between = np.empty((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            corrs_between[i, j] = np.nanmean(
                corr[
                    int(i * len(corr) / ndim) : int((i + 1) * len(corr) / ndim),
                    int(j * len(corr) / ndim) : int((j + 1) * len(corr) / ndim),
                ]
                / corrs[i] ** 0.5
                / corrs[j] ** 0.5
            )

    return corrs, corrs_between
