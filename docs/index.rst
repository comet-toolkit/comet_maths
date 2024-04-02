Welcome to comet_maths's documentation!
=========================================
**comet_maths** is a python module with useful mathematical algorithms for general use as well as for the other tools in the `CoMet Toolkit <https://www.comet-toolkit.org>`_.

There are quite a range of different functionalities within **comet_maths**. There are currently three submodules. One for linear algebra (mainly used for matrix operations in both `obsarray <https://obsarray.readthedocs.io/en/latest/>`_  and `punpy <https://punpy.readthedocs.io/en/latest/>`_ ), one for random sample generators (mainly used for sample generation in `punpy <https://punpy.readthedocs.io/en/latest/>`_ ) and one for interpolation (for general use).

The interpolation submodules focuses on two aspects. First, it aims to provide interpolation uncertainties which include both a contribution from the uncertainty on the input data point, as well as a contribution from the uncertainty in the model used for interpolation. Second, the interpolation module has functionality to interpolate between some low-resolution data points following a high resolution example. The example spectrum gets scaled in order to go through the low-resolution data points to form a sensible interpolation.

.. grid:: 2
    :gutter: 2

    .. grid-item-card::  Quickstart Guide
        :link: content/getting_started
        :link-type: doc

        New to *comet_maths*? Check out the quickstart guide for an introduction.

    .. grid-item-card::  User Guide
        :link: content/user_guide
        :link-type: doc

        The user guide provides a documentation and examples how to use **comet_maths** for interpolation, linear algebra or generating Monte Carlo samples.

    .. grid-item-card::  ATBD
        :link: content/atbd
        :link-type: doc

        ATBD mathematical description of **comet_maths** (under development).

    .. grid-item-card::  API Reference
        :link: content/API/api
        :link-type: doc

        The API Reference contains a description the **comet_maths** API.

Acknowledgements
----------------

**comet_maths** has been developed by `Pieter De Vis <https://github.com/pdevis>`_.

The development has been funded by:

* The UK's Department for Business, Energy and Industrial Strategy's (BEIS) National Measurement System (NMS) programme
* The IDEAS-QA4EO project funded by the European Space Agency.

Project status
--------------

**comet_maths** is under active development. It is beta software.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For users

   content/getting_started
   content/user_guide
   content/api
   content/atbd
