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

def expand_errcorr_dims(in_corr,in_dim,out_dim,dim_sizes):
    """
    Function to expand the provided correlation matrix (which defines the correlation along 1 or 2 dimensions),
    to higher dimensions, so that the total correlation matrix can be calculated.

    :param in_corr: correlation matrix along the dimensions specified in in_dim
    :type in_corr: np.ndarray
    :param in_dim: list of dimensions of the input correlation matrix
    :type in_dim: str or List[str]
    :param out_dim: list of dimensions of the expanded matrix
    :type out_dim: List[str]
    :param dim_sizes: dictionary with the dimensions of each of the dimensions
    :type dim_sizes: dict
    :return: correlation matrix contribution to the full correlation matrix
    :rtype: np.ndarray
    """
    #initialise the output correlation matrix
    totcorrlen = 1
    for i in range(len(out_dim)):
        totcorrlen *= dim_sizes[out_dim[i]]
    out_corr = np.eye(totcorrlen)

    loopshape=tuple([dim_sizes[dim] for dim in out_dim if dim not in in_dim])
    # if the input correlation is only along one dimension, do one of the three types of loop, depending if the correlation dimension is first, last or other in the output dimensions.
    if isinstance(in_dim,str):
        if in_dim==out_dim[0]:
            out_corr=first_dim_loop(in_corr,out_corr,loopshape)
        elif in_dim==out_dim[-1]:
            out_corr=last_dim_loop(in_corr,out_corr,loopshape,dim_sizes[out_dim[-1]])
        elif in_dim==out_dim[1]:
            for i,dim in enumerate(out_dim):
                if in_dim==out_dim[i]:
                    dimid=i
            out_corr=other_dim_loop(in_corr,out_corr,loopshape,dimid,out_dim,dim_sizes)
    # if the input correlation matrix is a list with one element, convert this dimension to string and restart
    elif len(in_dim)==1:
        expand_errcorr_dims(in_corr,in_dim[0],out_dim,dim_sizes)
    #if the input correlation matrix is along 2 dimensions, check if these dimensions are the first two or last two, and use similar approach as above
    elif len(in_dim)==2:
        if (in_dim[0]==out_dim[0] and in_dim[1]==out_dim[1]):
            out_corr=first_dim_loop(in_corr,out_corr,loopshape)
        elif (in_dim[1]==out_dim[0] and in_dim[0]==out_dim[1]):
            #in case the in in_corr dimensions are not in the same order as the out_dim, the in_corr is first reordered
            in_corr_flipped=change_order_errcorr_dims(in_corr,in_dim,[in_dim[1],in_dim[0]],dim_sizes)
            out_corr=first_dim_loop(in_corr_flipped,out_corr,loopshape)
        elif (in_dim[0]==out_dim[-2] and in_dim[1]==out_dim[-1]):
            out_corr=last_dim_loop(in_corr,out_corr,loopshape,dim_sizes[out_dim[-1]]*dim_sizes[out_dim[-2]])
        elif (in_dim[1]==out_dim[-2] and in_dim[0]==out_dim[-1]):
            in_corr_flipped=change_order_errcorr_dims(in_corr,in_dim,[in_dim[1],in_dim[0]],dim_sizes)
            out_corr=last_dim_loop(in_corr_flipped,out_corr,loopshape,dim_sizes[out_dim[-1]]*dim_sizes[out_dim[-2]])
        else:
            raise ValueError("comet_maths.matrix_conversion: this type of 2D indim not yet supported")
    else:
        raise ValueError("comet_maths.matrix_conversion: this type of indim not yet supported")
    return out_corr

def first_dim_loop(in_corr,out_corr,loopshape):
    """
    Loop to expand the err_corr matrix to higher dimension(s) if the dimensions in the in_corr are the first in the out_corr

    :param in_corr: correlation matrix along the dimensions specified in in_dim
    :type in_corr: np.ndarray
    :param out_corr: initialised output correlation matrix
    :type out_corr: np.ndarray
    :param loopshape: shape of the loop which determined how many times in_corr needs to be copied (determined by the out_corr dimensions not in in_corr)
    :type loopshape: tuple
    :return: correlation matrix contribution to the full correlation matrix
    :rtype: np.ndarray
    """
    looplen=np.prod(loopshape)
    for ii,mi in enumerate(np.ndindex(loopshape)):
        idx_start=ii
        idx_end=len(out_corr)
        ids=slice(idx_start,idx_end,looplen)
        out_corr[ids,ids] = in_corr
    return out_corr

def last_dim_loop(in_corr,out_corr,loopshape,size_last):
    """
    Loop to expand the err_corr matrix to higher dimensions if the dimension(s) in the in_corr are the last in the out_corr

    :param in_corr: correlation matrix along the dimensions specified in in_dim
    :type in_corr: np.ndarray
    :param out_corr: initialised output correlation matrix
    :type out_corr: np.ndarray
    :param loopshape: shape of the loop which determined how many times in_corr needs to be copied (determined by the out_corr dimensions not in in_corr)
    :type loopshape: tuple
    :param size_last: combined length of the last dimension(s)
    :type size_last: int
    :return: correlation matrix contribution to the full correlation matrix
    :rtype: np.ndarray
    """
    for ii,mi in enumerate(np.ndindex(loopshape)):
        idx_start=ii*size_last
        idx_end=(ii+1)*size_last
        ids=slice(idx_start,idx_end)
        out_corr[ids,ids] = in_corr
    return out_corr

def other_dim_loop(in_corr,out_corr,loopshape,other_dim_id,out_dim,dim_sizes):
    """
    Loop to expand the err_corr matrix to higher dimensions if the dimension(s) in the in_corr are not first and not last in the out_corr

    :param in_corr: correlation matrix along the dimensions specified in in_dim
    :type in_corr: np.ndarray
    :param out_corr: initialised output correlation matrix
    :type out_corr: np.ndarray
    :param loopshape: shape of the loop which determined how many times in_corr needs to be copied (determined by the out_corr dimensions not in in_corr)
    :type loopshape: tuple
    :param other_dim_id: index of the in_corr dimension in the out_dim list
    :type other_dim_id: int
    :param out_dim: list of dimensions of the expanded matrix
    :type out_dim: List[str]
    :param dim_sizes: dictionary with the dimensions of each of the dimensions
    :type dim_sizes: dict
    :return: correlation matrix contribution to the full correlation matrix
    :rtype: np.ndarray
    """
    for ii,mi in enumerate(np.ndindex(loopshape)):
        small_size=np.prod([dim_sizes[dim] for dim in out_dim[other_dim_id+1:]])
        big_size=np.prod([dim_sizes[dim] for dim in out_dim[other_dim_id:]])
        idx_start=ii%small_size+(ii//small_size)*big_size
        idx_end=ii%small_size+(ii//small_size)*big_size+len(in_corr)*small_size
        ids=slice(idx_start,idx_end,small_size)
        out_corr[ids,ids] = in_corr
    return out_corr



def change_order_errcorr_dims(in_corr,in_dim,out_dim,dim_sizes):
    """
    Function to flip the order of the underlying dimensions for an err_corr for matrices that describe the combination of 2 dimensions

    :param in_corr: correlation matrix along the dimensions specified in in_dim
    :type in_corr: np.ndarray
    :param in_dim: list of dimensions of the input correlation matrix
    :type in_dim: str or List[str]
    :param out_dim: list of dimensions of the expanded matrix
    :type out_dim: List[str]
    :param dim_sizes: dictionary with the dimensions of each of the dimensions
    :type dim_sizes: dict
    :return: correlation matrix with flipped underlying dimensions
    :rtype: np.ndarray
    """
    if not len(in_dim)==len(out_dim):
        raise ValueError("comet_maths.matrix_conversion: in_dim and out_dim should have the same length")
    elif isinstance(in_dim,str):
        raise ValueError("comet_maths.matrix_conversion: in_dim and out_dim should be list of dimension, not a single string")
    elif in_dim==out_dim:
        return in_corr
    else:
        out_corr=np.eye(len(in_corr))
        if len(in_dim)==2:
            for i in range(len(out_corr)):
                for j in range(len(out_corr)):
                    jj=i//dim_sizes[in_dim[0]]
                    ii=i%dim_sizes[in_dim[0]]
                    i2=jj+ii*dim_sizes[out_dim[0]]
                    jj2=j//dim_sizes[in_dim[0]]
                    ii2=j%dim_sizes[in_dim[0]]
                    j2=jj2+ii2*dim_sizes[out_dim[0]]
                    out_corr[i,j]=in_corr[i2,j2]
        else:
            raise ValueError("comet_maths.matrix_conversion: currently only matrices with 2 in_dim can be flipped")
    return out_corr

