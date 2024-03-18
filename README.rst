comet_maths
===========

Mathematical algorithms and tools to use within CoMet toolkit.

Usage
=====

Virtual environment
-------------------

It's always recommended to make a virtual environment for each of your python
projects. Use your preferred virtual environment manager if you want and
activate it for the rest of these commands.

If you are using conda you can create and activate your environment using::

    conda create -n yourenvname -k python=3.x

followed by::

    conda activate yourenvname (activate environment in windows)

or::

    source activate yourenvname (activate environment on a UNIX operating system)

You can also use venv. If you're unfamiliar, read
https://realpython.com/python-virtual-environments-a-primer/. You can set one up
using::

    python -m venv venv

and then activate it on Windows by using ``venv/Scripts/activate``. 

Installation
------------

Install your package and its dependancies by using::

    pip install -e .

Development
-----------

For developing the package, you'll want to install the pre-commit hooks as well. Type::

    pre-commit install


Note that from now on when you commit, `black` will check your code for styling
errors. If it finds any it will correct them, but the commit will be aborted.
This is so that you can check its work before you continue. If you're happy,
just commit again. 

Running the test suite is possible by running `pytest`.

Compatibility
-------------

Licence
-------

Authors
-------

`comet_maths` was written by `Pieter De Vis <pieter.de.vis@npl.co.uk>`_.
