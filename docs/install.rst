Installation
============

Installation via PIP
--------------------
The easiest way to install skultrafast is using *pip*. Since skultrafast is a
pure python package, no compiler is necessary.

To get the latest released version from pypi::

    pip install skultrafast

To get the latest development version from GitHub (requires git installed)::

    pip install git+https://github.com/tillsten/skultrafast --upgrade


If you are using anaconda, git can be installed by::

    conda install git


Software prerequisites
----------------------
Needs Python 3.6 or higher. Requires the following packages,
which are automatically installed when using pip:

* numpy
* numba
* scipy
* lmfit
* matplotlib
* sklearn
* sympy

To build the docs, more packages are necessary.

* sphinx
* sphinx-gallery
