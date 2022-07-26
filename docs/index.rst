Welcome to comet_maths's documentation!
=========================================================
**comet_maths** is a python module with useful mathematical algorithms for general use as well as for the other tools in the comet toolkit.

There are quite a range of different functionalities within *comet_maths*. There are currently three submodules. One for linear algebra (mainly used for matrix operations in both *obsarray* and *punpy*), one for random generators (mainly used for sample generation in *punpy*) and one for interpolation (for general use).

The interpolation submodules focuses on two aspects. First, it aims to provide interpolation uncertainties that are as realistic as possible, and include both a contribution from the uncertainty on the input data point, as well as a contribution from the uncertainty in the model used for interpolation. Second, the interpolation module has functionality to interpolate between some low-resolution data points following a high resolution example. The example spectrum gets scaled in order to go through the low-resolution data points to form a sensible interpolation.

~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   content/getting_started
   content/linear_algebra
   content/random_generator
   content/interpolation
   content/examples
   content/atbd


API Documentation
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 4

   content/API/comet_maths

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
