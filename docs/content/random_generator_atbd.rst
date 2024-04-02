.. Overview of method
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/22

.. _random_generator_atbd:

================================================
Random Generator Algorithm Theoretical Basis
================================================

The aim of the random generator in comet_maths is to generate samples of draws from a given Probability Density Function (PDF).
The standard Probability Density Function (PDF) is Gaussian, but other PDF are available
too (see :ref:`pdf`). For Gaussian PDF, comet_maths generates samples of draws (total number of
draws is set by keyword `MCsteps`) based on the provided input quantities (the mean of the PDF) and provided uncertainties (the width of the PDF).

Internally, comet_maths always generates independent random normally distributed
samples first and then correlates them where necessary using the Cholesky decomposition method (see paragraph below).
Using this Cholesky decomposition correlates the PDF of the input quantities which means the joint PDF are defined.

Cholesky decomposition is a useful method from linear algebra, which allows to efficiently draw samples from a
multivariate probability distribution (joint PDF). The Cholesky decomposition is a decomposition of a
positive-definite matrix into the product of a lower triangular matrix and its conjugate transpose. The positive-definite
matrix being decomposed here is the correlation or covariance matrix (S(X)) and R is the upper triangular matrix given by the
Cholesky decomposition:

:math:`S(X)=R^T R`.

When sampling from the joint pdf, one can first draw samples :math:`Z = (Z_{i},\ldots,\ Z_{N})` for the input quantities :math:`X_i` from the
independent PDF for the input quantities (i.e. as if they were uncorrelated). These samples :math:`Z_i` can then be combined
with the decomposition matrix R to obtain the correlated samples :math:`\xi = (\xi_1, ... , \xi_N)`:

:math:`\xi = X + R^T Z`.