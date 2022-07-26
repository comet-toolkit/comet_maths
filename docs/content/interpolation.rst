.. Overview of method
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _interpolation:

=======================================
Comet_maths Interpolation Module
=======================================

In this section we explain how **comet_maths** can be
used to interpolate between different datapoints.
The **comet_maths** interpolation module can be used to
propagate uncertainties through the interpolation process.
The uncertainty on the interpolated data points includes
contributions both from the uncertainties on the input data
points, as well as uncertainties on the interpolation process itself.

There are currently two main use cases for which the **comet_maths**
interpolation module can be used:
-  **normal 1D interpolation**: This is the typical use case
for interpolation. The added value of the tool is that it allows
to automatically determine the uncertainties (and error-correlation information)
on the interpolated datapoints. The following methods can be used: "linear",
"quadratic", "cubic", "nearest", "next", "previous", "lagrange", "ius" and "gpr".
The latter two stand for Interpolated Univariate Spline and Gaussian Process
Regression. For the implementation of the interpolation methods themselves,
we wrap the sklearn (for gpr) and scipy (for all others) implementation,
and add uncertainty propagation to it.

-  **1D interpolation following a high-resolution example**: For this use case,
we have developed a method so that an interpolation can be done between a set of
low resolution data points, which follows the shape of another set of high-resolution
data points (or a high resolution model). This is particularly useful if the
low-resolution data points have small uncertainties (and thus a good absolute calibration),
yet the shape of the function between the data points is known from another set of data
points or model which have larger uncertainties (e.g. if we have a set of measurements
with poor absolute calibration but good relative calibration). By combining these measurements,
the high-resolution model can be confidently anchored to the low-resolution datapoints, thereby
significantly reducing its uncertainties (especially any error-contributions that are systematic).
In short, this method works by first calculating a set of residuals between the
low-resolution data and high-resolution data, then interpolating these residuals
to all requested wavelengths, and then combining them again with the high resolution
datapoints in order to get the final interpolated datapoints. The residuals can be
calculated in a relative or absolute way (depending on which is most appropriate
for a given use case). The interpolation steps throughout this method use the same
interpolation methods and uncertainty propagation as the **normal 1D interpolation**.


Usage
#######################
