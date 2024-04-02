.. Overview of method
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _linear_algebra:

=======================================
Comet_maths Linear Algebra Module
=======================================

The comet_maths linear algebra module implements matrix operations, typically applied to error correlation and covariance matrices.
Most of these are used within punpy and obsarray. We will here go through the most important functions.

First, there is function to calulate the error correlation matrix from a MC-generated sample::

   import comet_maths as cm
   import numpy as np
   sample = cm.generate_sample(1000, np.zeros(200), np.ones(200), "syst")
   err_corr = cm.calculate_corr(sample, corr_dims=0)

Here it is possible to provide the dimension (or list of dimensions) along which to calculate the error correlation.
If no corr_dims is provided, all dimensions will be used (can lead to memory intensive operation if dimensions are large).


Another important function that is often used (any time when generating a correlated sample) is the nearestPD_cholesky().
This function checks if the provided (error correlation/covariance) matrix is positive definite, and if not modifies
the diagonal elements of the matrix to try and make it positive definite (e.g. needed for fully systematic error correlation
matrices). The corr argument is a boolean which allows to indicate whether the provided matrix is an error correlation matrix.
If the latter is the case, we ensure there are no elements above 1. This function returns either the positive definite error
correlation matrix, or its Cholesky decomposition::

   A_cholesky = cm.nearestPD_cholesky(err_corr, corr=True)
   A_PD = cm.nearestPD_cholesky(err_corr, corr=True, return_cholesky=False)

The Cholesky decomposition is useful to correlate samples (see :ref:`random_generator_atbd`).


Next there is also a function to calculate the Jacobian of a measurement function::

   Jx = cm.calculate_Jacobian(function_flat, xflat)

Here the input quantities need to be flattened into a single array, and the measurement function should run with
these flattened input quantities.
Typically, this requires concatenating the input quantities, and making a new measurement function which indexes
the flattened input quantities to get back to the individual variables before applying the rest of the measurement function.


Then, there are a range of functions which allow to convert an error correlation matrix and associated uncertainties into a covariance matrix and the other way around::

   errcorrx = cm.correlation_from_covariance(cov_x)
   u_x = cm.uncertainty_from_covariance(cov_x)
   errcorrx = cm.convert_cov_to_corr(cov_x, u_x)
   cov_x = cm.convert_corr_to_cov(errcorrx, u_x)

There are also some functions which allow to combine different error correlation matrices
for which the variables themselves are corelated with error correlation matrix corr_between,
as well as functions to separate a combined matrix::

   corr_between=np.array([[1, 0.4], [0.4, 1]])
   flat_corr = cm.calculate_flattened_corr(
            [errcorrx, 0.5*np.ones_like(errcorrx)+0.5 * np.eye(len(errcorrx))],
            corr_between
        )
   corrs, corr_between = cm.separate_flattened_corr(flat_cor, 2) # here the 2 is how many error correlation arrays the provided array should be splitted.

It is important to note that here the error correlation matrices provided (or separated into) must be of the same shape.
In addition to taking into account the corr_between values, to calculate the error correlation between e.g. the error on the first index of the first variable
and the third index of the second variable, we average the error-correlation between the first and third
index of the first variable and the error correlation between the first and third index of the second variable.

There are also other useful functions that are used within punpy, but might also be useful by themselves (though these should be used with caution).
This includes e.g. a function to change the order of variable dimensions and calculate the resulting changes to the error correlation matrix::

   errcorryx2 = cm.change_order_errcorr_dims(
            errcorrxy, ["x", "y"], ["y", "x"], dim_sizes
        )

Another function allows to expand error correlation matrices in one or more dimensions to more dimensions::

   outcorr2_x = cm.expand_errcorr_dims(errcorrx, "x", ["x", "y", "z"], dim_sizes)
   outcorr2_y = cm.expand_errcorr_dims(errcorry, "y", ["x", "y", "z"], dim_sizes)
   outcorr2_z = cm.expand_errcorr_dims(errcorrz, "z", ["x", "y", "z"], dim_sizes)

   outcorr2_xyz = np.dot(np.dot(outcorr2_x, outcorr2_y), outcorr2_z)

This is typically needed when calculating the total error correlation matrix from error correlation matrices provided
separately per dimension. The expanded error correlation matrix returned by this function should be used
in a dot product with error correlation matrices in other dimensions (the expanded matrix is not meaningful by itself).
