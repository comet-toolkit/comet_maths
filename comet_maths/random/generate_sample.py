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
    elif not hasattr(x[i], "__len__"):
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


def generate_sample_correlated(MCsteps, x, u_x, corr_x, i, dtype=None):
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
    if x[i].ndim == 2:
        if len(corr_x[i]) == len(u_x[i]):
            MC_data = np.zeros((u_x[i].shape) + (MCsteps,))
            for j in range(len(u_x[i][0])):
                cov_x = cm.convert_corr_to_cov(corr_x[i], u_x[i][:, j])
                MC_data[:, j, :] = generate_sample_cov(
                    MCsteps, x[i][:, j].flatten(), cov_x, dtype=dtype
                ).reshape(x[i][:, j].shape + (MCsteps,))
        else:
            MC_data = np.zeros((u_x[i].shape) + (MCsteps,))
            for j in range(len(u_x[i][:, 0])):
                cov_x = cm.convert_corr_to_cov(corr_x[i], u_x[i][j])
                MC_data[j, :, :] = generate_sample_cov(
                    MCsteps, x[i][j].flatten(), cov_x, dtype=dtype
                ).reshape(x[i][j].shape + (MCsteps,))
    else:
        cov_x = cm.convert_corr_to_cov(corr_x[i], u_x[i])
        MC_data = generate_sample_cov(
            MCsteps, x[i].flatten(), cov_x, dtype=dtype
        ).reshape(x[i].shape + (MCsteps,))

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
    MC_sample = np.tile(param, (MCsteps,) + (1,) * param.ndim)
    MC_sample = np.moveaxis(MC_sample, 0, -1)
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

    elif len(param.shape) == 1:
        return (
            np.random.normal(size=(len(param), MCsteps)).astype(dtype)
            * u_param[:, None]
            + param[:, None]
        )
    elif len(param.shape) == 2:
        return (
            np.random.normal(size=param.shape + (MCsteps,)) * u_param[:, :, None]
            + param[:, :, None]
        )
    elif len(param.shape) == 3:
        return (
            np.random.normal(size=param.shape + (MCsteps,)).astype(dtype)
            * u_param[:, :, :, None]
            + param[:, :, :, None]
        )
    elif len(param.shape) == 4:
        return (
            np.random.normal(size=param.shape + (MCsteps,)).astype(dtype)
            * u_param[:, :, :, :, None]
            + param[:, :, :, :, None]
        )
    else:
        raise ValueError(
            "punpy.mc_propagation: parameter shape not supported: %s %s"
            % (param.shape, param)
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
    elif len(param.shape) == 1:
        return (
            np.dot(
                u_param[:, None],
                np.random.normal(size=MCsteps).astype(dtype)[None, :],
            )
            + param[:, None]
        )
    elif len(param.shape) == 2:
        return (
            np.dot(
                u_param[:, :, None],
                np.random.normal(size=MCsteps).astype(dtype)[:, None, None],
            )[:, :, :, 0]
            + param[:, :, None]
        )
    elif len(param.shape) == 3:
        return (
            np.dot(
                u_param[:, :, :, None],
                np.random.normal(size=MCsteps).astype(dtype)[:, None, None, None],
            )[:, :, :, :, 0, 0]
            + param[:, :, :, None]
        )
    elif len(param.shape) == 4:
        return (
            np.dot(
                u_param[:, :, :, :, None],
                np.random.normal(size=MCsteps).astype(dtype)[:, None, None, None, None],
            )[:, :, :, :, :, 0, 0, 0]
            + param[:, :, :, :, None]
        )
    else:
        raise ValueError(
            "punpy.mc_propagation: parameter shape not supported: %s %s"
            % (param.shape, param)
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

    if dtype is None:
        return np.dot(L, np.random.normal(size=(len(param), MCsteps))) + param[:, None]
    else:
        return (
            np.dot(L, np.random.normal(size=(len(param), MCsteps)).astype(dtype))
            + param[:, None]
        )


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
        for j in np.ndindex(sample[0][..., 0].shape):
            sample_j = np.array([sample[i][j] for i in range(len(sample))], dtype=dtype)

            # Cholesky needs to be applied to Gaussian distributions with mean=0 and std=1,
            # We first calculate the mean and std for each input quantity
            means = np.array(
                [np.mean(sample[i][j]) for i in range(len(sample))],
                dtype=dtype,
            )[:, None]
            stds = np.array(
                [np.std(sample[i][j]) for i in range(len(sample))],
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
                sample_out[i][j] = sample_j[i]

        return sample_out
