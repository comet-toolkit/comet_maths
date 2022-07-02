"""Class with methods that convert a covariance or correlation matrix into a related metric"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import warnings
from typing import List, Tuple
import numpy as np

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix

    :param covariance: Covariance matrix
    :return: Correlation matrix
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def uncertainty_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix to uncertainty

    :param covariance: Covariance matrix
    :return: uncertainties
    """
    return np.sqrt(np.diag(covariance))


def convert_corr_to_cov(corr: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to covariance matrix

    :param corr: correlation matrix
    :param u: uncertainties
    :return: covariance matrix
    """
    return u.reshape((-1, 1)) * corr * (u.reshape((1, -1)))


def convert_cov_to_corr(cov: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix

    :param corr: covariance matrix
    :param u: uncertainties
    :return: correlation matrix
    """
    return 1 / u.reshape((-1, 1)) * cov / (u.reshape((1, -1)))


def calculate_flattened_corr(corrs: List[np.ndarray], corr_between: np.ndarray) -> np.ndarray:
    """
    Combine correlation matrices for different input quantities, with a correlation
    matrix that gives the correlation between the input quantities into a full
    (flattened) correlation matrix combining the two.

    :param corrs: list of correlation matrices for each input quantity
    :param corr_between: correlation matrix between the input quantities
    :return: full correlation matrix combining the correlation matrices
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


def separate_flattened_corr(corr: np.ndarray, ndim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate a full (flattened) correlation matrix into a list of correlation matrices
    for each output variable and a correlation matrix between the output variables.

    :param corr: full correlation matrix
    :param ndim: number of output variables
    :return: list of correlation matrices for each output variable, correlation matrix between the output variables
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
