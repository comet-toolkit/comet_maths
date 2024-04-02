.. currentmodule:: comet_maths

.. _api:

#############
API reference
#############

This page provides an auto-generated summary of **comet_maths**'s API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.

Interpolation
====================

.. autosummary::
   :toctree: generated/

   interpolation.interpolation.Interpolator
   interpolation.interpolation.Interpolator.interpolate_1d
   interpolation.interpolation.Interpolator.interpolate_1d_along_example
   interpolation.interpolation.interpolate_1d
   interpolation.interpolation.default_unc_methods
   interpolation.interpolation.model_error_analytical_methods
   interpolation.interpolation.gaussian_process_regression
   interpolation.interpolation.gpr_basics
   interpolation.interpolation.interpolate_1d_along_example

Linear Algebra
=====================

.. autosummary::
   :toctree: generated/

   linear_algebra.matrix_calculation.calculate_Jacobian
   linear_algebra.matrix_calculation.calculate_corr
   linear_algebra.matrix_calculation.nearestPD_cholesky
   linear_algebra.matrix_calculation.isPD
   linear_algebra.matrix_conversion.correlation_from_covariance
   linear_algebra.matrix_conversion.uncertainty_from_covariance
   linear_algebra.matrix_conversion.convert_corr_to_cov
   linear_algebra.matrix_conversion.convert_cov_to_corr
   linear_algebra.matrix_conversion.calculate_flattened_corr
   linear_algebra.matrix_conversion.separate_flattened_corr
   linear_algebra.matrix_conversion.expand_errcorr_dims
   linear_algebra.matrix_conversion.change_order_errcorr_dims

Generating MC Samples
=======================

.. autosummary::
   :toctree: generated/

   random.generate_sample.generate_sample
   random.generate_sample.generate_error_sample
   random.generate_sample.generate_sample_shape
   random.generate_sample.generate_sample_same
   random.generate_sample.generate_sample_random
   random.generate_sample.generate_sample_systematic
   random.generate_sample.generate_sample_correlated
   random.generate_sample.generate_sample_corr
   random.generate_sample.generate_sample_cov
   random.generate_sample.correlate_sample_corr
   random.probability_density_function.generate_sample_pdf
