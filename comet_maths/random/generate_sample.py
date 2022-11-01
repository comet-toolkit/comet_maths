"""describe class"""

"""___Built-In Modules___"""
import comet_maths as cm

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def generate_sample(MCsteps, x, u_x, corr_x, i=None, dtype=None):
    """
    Generate correlated MC sample of input quantity with given uncertainties and correlation matrix.

    :param x: list of input quantities (usually numpy arrays)
    :type x: list[array]
    :param u_x: list of uncertainties/covariances on input quantities (usually numpy arrays)
    :type u_x: list[array]
    :param corr_x: list of correlation matrices (n,n) along non-repeating axis, or list of correlation matrices for each repeated measurement.
    :type corr_x: list[array]
    :param i: index of the input quantity (in x)
    :type i: int, optional
    :param dtype: dtype of the produced sample, optional
    :type dtype: numpy.dtype
    :return: generated sample
    :rtype: array
    """
    if i is None:
        x = np.array([x])
        u_x = np.array([u_x])
        corr_x = np.array([corr_x])
        i = 0
    if np.count_nonzero(u_x[i]) == 0:
        sample = generate_sample_same(MCsteps, x[i], dtype=dtype)
    elif not hasattr(x[i], "size"):
        sample = generate_sample_systematic(MCsteps, x[i], u_x[i], dtype=dtype)
    elif x[i].size == 1:
        sample = generate_sample_systematic(MCsteps, x[i], u_x[i], dtype=dtype)
    elif isinstance(corr_x[i], str):
        if corr_x[i] == "rand":
            sample = generate_sample_random(MCsteps, x[i], u_x[i], dtype=dtype)
        elif corr_x[i] == "syst":
            sample = generate_sample_systematic(MCsteps, x[i], u_x[i], dtype=dtype)
    else:
        sample = generate_sample_correlated(MCsteps, x, u_x, corr_x, i, dtype=dtype)

    if MCsteps == 1:
        sample = sample.squeeze()
    return sample


def generate_sample_correlated(MCsteps, x, u_x, corr_x, i=None, dtype=None):
    """
    Generate correlated MC sample of input quantity with given uncertainties and correlation matrix.
    sample are generated using generate_sample_cov() after matching up the uncertainties to the right correlation matrix.
    It is possible to provide one correlation matrix to be used for each measurement (which each have an uncertainty) or a correlation matrix per measurement.

    :param x: list of input quantities (usually numpy arrays)
    :type x: list[array]
    :param u_x: list of uncertainties/covariances on input quantities (usually numpy arrays)
    :type u_x: list[array]
    :param corr_x: list of correlation matrices (n,n) along non-repeating axis, or list of correlation matrices for each repeated measurement.
    :type corr_x: list[array]
    :param i: index of the input quantity (in x)
    :type i: int
    :param dtype: dtype of the produced sample, optional
    :type dtype: numpy.dtype
    :return: generated sample
    :rtype: array
    """

    if i is None:
        x = np.array([x])
        u_x = np.array([u_x])
        corr_x = np.array([corr_x])
        i = 0

    if corr_x[i].ndim == 2:
        if len(corr_x[i]) == len(u_x[i].ravel()):
            cov_x = cm.convert_corr_to_cov(corr_x[i], u_x[i])
            MC_data = generate_sample_cov(
                MCsteps, x[i], cov_x, dtype=dtype
            )
        elif len(corr_x[i]) == len(u_x[i]):
            MC_data = np.zeros((MCsteps,)+(u_x[i].shape))
            for j in range(len(u_x[i][0])):
                cov_x = cm.convert_corr_to_cov(corr_x[i], u_x[i][:, j])
                MC_data[:, :, j] = generate_sample_cov(
                    MCsteps, x[i][:, j], cov_x, dtype=dtype
                )
        elif len(corr_x[i]) == len(u_x[i][0]):
            MC_data = np.zeros((MCsteps,)+(u_x[i].shape))
            for j in range(len(u_x[i][:, 0])):
                cov_x = cm.convert_corr_to_cov(corr_x[i], u_x[i][j])
                MC_data[:, j, :] = generate_sample_cov(
                    MCsteps, x[i][j], cov_x, dtype=dtype
                )
        else:
            raise NotImplementedError(
                "comet_maths.generate_Sample: This combination of dimension of correlation matrix (%s) and uncertainty (%s) is currently not implemented."
                % (corr_x[i].shape, u_x[i].shape)
            )

    elif (len(corr_x[i])==x[i].ndim):
        if np.all([len(corr_x[i][j])==x[i].shape[j] for j in range(x[i].ndim)]):
            MC_data=generate_sample_random(MCsteps, x[i], u_x[i], dtype=dtype)
            for j in range(x[i].ndim):
                MC_data=correlate_sample_corr(MC_data, corr_x[i][j])
        else:
            ValueError("When providing a list of correlation matrices for each dimension, ")
    else:
        raise NotImplementedError(
            "comet_maths.generate_Sample: This combination of dimension of correlation matrix (%s) and uncertainty (%s) is currently not implemented."
            % (corr_x[i].shape, u_x[i].shape)
        )

    return MC_data


def generate_sample_same(MCsteps, param, dtype=None):
    """
    Generate MC sample of input quantity with zero uncertainties.

    :param param: values of input quantity (mean of distribution)
    :type param: float or array
    :param dtype: dtype of the produced sample, optional
    :type dtype: numpy.dtype
    :return: generated sample
    :rtype: array
    """
    if isinstance(param, np.ndarray):
        tileshape = (MCsteps,) + (1,) * param.ndim
    else:
        tileshape = (MCsteps,)
    MC_sample = np.tile(param, tileshape).astype(dtype)
    return MC_sample


def generate_sample_random(MCsteps, param, u_param, dtype=None):
    """
    Generate MC sample of input quantity with random (Gaussian) uncertainties.

    :param param: values of input quantity (mean of distribution)
    :type param: float or array
    :param u_param: uncertainties on input quantity (std of distribution)
    :type u_param: float or array
    :param dtype: dtype of the produced sample, optional
    :type dtype: numpy.dtype
    :return: generated sample
    :rtype: array
    """
    if not hasattr(param, "__len__"):
        return np.random.normal(size=MCsteps).astype(dtype) * u_param + param
    elif len(param.shape) == 0:
        return np.random.normal(size=MCsteps).astype(dtype) * u_param + param
    else:
        sli_par=list([slice(None)] * (len(param.shape) +1))
        sli_par[0]=None

        return (
                np.random.normal(size=(MCsteps,)+param.shape).astype(dtype)
                * u_param[sli_par]
                + param[sli_par]
        )


def generate_sample_systematic(MCsteps, param, u_param, dtype=None):
    """
    Generate correlated MC sample of input quantity with systematic (Gaussian) uncertainties.

    :param param: values of input quantity (mean of distribution)
    :type param: float or array
    :param u_param: uncertainties on input quantity (std of distribution)
    :type u_param: float or array
    :param dtype: dtype of the produced sample, optional
    :type dtype: numpy.dtype
    :return: generated sample
    :rtype: array
    """
    if not hasattr(param, "__len__"):
        return np.random.normal(size=MCsteps).astype(dtype) * u_param + param
    elif len(param.shape) == 0:
        return np.random.normal(size=MCsteps).astype(dtype) * u_param + param
    else:
        sli_par=list([slice(None)] * (len(param.shape) +1))
        sli_par[-2]=None

        return (
            np.dot(
                np.random.normal(size=MCsteps).astype(dtype)[:, None],
                u_param[sli_par],
            )
            + param
        )

def generate_sample_cov(MCsteps, param, cov_param, dtype=None, diff=0.01):
    """
    Generate correlated MC sample of input quantity with a given covariance matrix.
    sample are generated independent and then correlated using Cholesky decomposition.

    :param param: values of input quantity (mean of distribution)
    :type param: array
    :param cov_param: covariance matrix for input quantity
    :type cov_param: array
    :param dtype: dtype of the produced sample, optional
    :type dtype: numpy.dtype
    :return: generated sample
    :rtype: array
    """
    try:
        L = np.linalg.cholesky(cov_param)
    except:
        L = cm.nearestPD_cholesky(cov_param, diff=diff)

    outshape=param.shape

    if param.ndim>1:
        param=param.flatten()

    if len(param)!=len(L):
        raise ValueError("The shapes of the provided variable (%s after flattening) and the provided covariance matrix (%s) are not consistent"%(param.shape,L.shape))

    return (
            np.dot(L, np.random.normal(size=(len(L),MCsteps))).astype(dtype).T
            + param
        ).reshape((MCsteps,)+outshape)


def correlate_sample_corr(sample, corr, dtype=None):
    """
    Method to correlate independent sample of input quantities using correlation matrix and Cholesky decomposition.

    :param sample: independent sample of input quantities
    :type sample: array[array]
    :param corr: correlation matrix between input quantities
    :type corr: array
    :return: correlated sample of input quantities
    :rtype: array[array]
    """

    if np.max(corr) > 1.000001 or len(corr) != len(sample):
        raise ValueError(
            "punpy.mc_propagation: The correlation matrix between variables is not the right shape or has elements >1."
        )
    else:
        try:
            L = np.array(np.linalg.cholesky(corr))
        except:
            L = cm.nearestPD_cholesky(corr)

        sample_out = sample.copy()
    for j in np.ndindex(sample[0][0, ...].shape):
        sample_j = np.array([sample[i][(slice(None),)+j] for i in range(len(sample))], dtype=dtype)

        # Cholesky needs to be applied to Gaussian distributions with mean=0 and std=1,
        # We first calculate the mean and std for each input quantity
        means = np.array(
            [np.mean(sample[i][(slice(None),)+j]) for i in range(len(sample))],
            dtype=dtype,
        )[:, None]
        stds = np.array(
            [np.std(sample[i][(slice(None),)+j]) for i in range(len(sample))],
            dtype=dtype,
        )[:, None]

        # We normalise the sample with the mean and std, then apply Cholesky, and finally reapply the mean and std.
        if all(stds[:, 0] != 0):
            sample_j = np.dot(L, (sample_j - means) / stds) * stds + means

        # If any of the variables has no uncertainty, the normalisation will fail. Instead we leave the parameters without uncertainty unchanged.
        else:
            id_nonzero = np.where(stds[:, 0] != 0)[0]
            sample_j[id_nonzero] = (
                    np.dot(
                        L[id_nonzero][:, id_nonzero],
                        (sample_j[id_nonzero] - means[id_nonzero]) / stds[id_nonzero],
                        )
                    * stds[id_nonzero]
                    + means[id_nonzero]
            )

        for i in range(len(sample)):
            sample_out[i][(slice(None),)+j] = sample_j[i]

    return sample_out