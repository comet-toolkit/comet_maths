"""comet_maths - Mathematical algorithms and tools to use within CoMet toolkit."""
from comet_maths.random.generate_sample import (
    generate_sample,
    generate_error_sample,
    generate_sample_systematic,
    generate_sample_random,
    generate_sample_cov,
    correlate_sample_corr,
)

from comet_maths.linear_algebra.matrix_calculation import (
    calculate_Jacobian,
    calculate_corr,
    nearestPD_cholesky,
    isPD,
)
from comet_maths.linear_algebra.matrix_conversion import (
    calculate_flattened_corr,
    separate_flattened_corr,
    convert_corr_to_cov,
    convert_cov_to_corr,
    correlation_from_covariance,
    uncertainty_from_covariance,
    change_order_errcorr_dims,
    expand_errcorr_dims,
)

from comet_maths.interpolation.interpolation import (
    interpolate_1d,
    interpolate_1d_along_example,
    Interpolator,
    gaussian_process_regression,
)

__author__ = "Pieter De Vis <pieter.de.vis@npl.co.uk>"
__all__ = []

from ._version import __version__
