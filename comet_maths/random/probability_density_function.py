"""describe class"""
import warnings

"""___Built-In Modules___"""
import comet_maths as cm

"""___Third-Party Modules___"""
import numpy as np
import numpy.random as rn

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def generate_sample_pdf(size, pdf_shape, pdf_params=None, dtype=None, seed=12345):
    """
    Function to generate samples from standard probability functions (with zero as mean and 1 as width)

    :param size: Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
    :type size: int or tuple of ints
    :param pdf_shape: string identifier of the probability density function shape, defaults to gaussian
    :type pdf_shape: str, optional
    :param pdf_params: dictionaries defining optional additional parameters that define the probability density function, Defaults to None (gaussian does not require additional parameters)
    :type pdf_params: dict, optional
    :param dtype: dtype of the output sample
    :type dtype: numpy.dtype, optional
    :return: output sample of given size and probability density function
    :rtype: array
    """
    if pdf_shape.lower() == "gaussian" or pdf_shape.lower() == "truncated_gaussian":
        return (rn.standard_normal(size=size)).astype(dtype)
    elif pdf_shape.lower() == "tophat":
        return (rn.uniform(size=size, low=-1.0, high=1.0)).astype(dtype)
    else:
        raise NotImplementedError("pdf shape (%s) not implemented" % (pdf_shape))
