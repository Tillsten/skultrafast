
.. image:: https://github.com/Tillsten/skultrafast/raw/master/docs/_static/skultrafast_logo_v1.svg
    :alt: ***skultrafast***

.. image:: https://readthedocs.org/projects/skultrafast/badge/?version=latest
    :target: https://skultrafast.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/Tillsten/skultrafast/workflows/Python%20package/badge.svg
    :target: https://github.com/Tillsten/skultrafast/actions?query=workflow%3A%22Python+package%22

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5713589.svg
    :target: https://doi.org/10.5281/zenodo.5713589

What is skultrafast?
--------------------
skultrafast is a Python package for ultrafast spectroscopy data analysis. It
provides tools for analyzing various types of time-resolved spectroscopy data,
including pump-probe, transient absorption, and two-dimensional infrared (2D-IR)
spectroscopy. The package includes functionality for data import, visualization,
processing, and fitting. It is built on top of the scientific Python ecosystem,
including NumPy, SciPy, and Matplotlib.

The latest version of the package is available on `github <https://github
.com/Tillsten/skultrafast>`_. A build of the documentation can be found at `Read
the docs <https://skultrafast.readthedocs.io/en/latest/>`_. The documentation
includes `Installtion notes
<https://skultrafast.readthedocs.io/en/latest/install.html>`_.


Funding
-------
The package was created and is maintained by *Till Stensitzki*. The package was
created while being employed in the `Heyne group <http://www.physik.fu-berlin
.de/einrichtungen/ag/ag-heyne/>`_ and was therefore founded by the DFG via `SFB
1078 <www.sfb1078.de/>`_ and `SFB 1114 <www.sfb1114.de/>`_. Recent development
focussed on 2D-spectroscopy is part of my stay in the `Ultrafast Structual Dynamics
Group <https://www.uni-potsdam.de/usd>`_ in Potsdam under Müller-Werkmeister.


Scope of the project
--------------------

I like to include any kind of algorithm or data structure which comes up in
ultrafast spectropy. Since I am mostly doing pump-probe and 2D-IR spectroscopy,
the current focus is on these two techniques. However, I am open to include any
other technique.

I am also open to add a graphical interface to the package,
but as experience shows, a GUI brings in a lot of maintenance burden. Hence, the
first target is a interactive data-explorer for the jupyter notebook.

This package also tries its best to follow modern software practices. This
includes version control using *git*, continues integration testing via
github action and a decent documentation hosted on `Read the docs`.

Features
--------
The current releases centers around working with time-resolved spectra:

  *  Generate publication-ready plots with minimal effort.
  *  Perform global fitting of transient data, including DAS, SAS, and
     compartment modeling.
  *  Analyze polarization-resolved datasets.
  *  Easily process data by selecting, filtering, and recalibrating it.
  *  Correct dispersion automatically in the case of chirped spectra.
  *  Obtain accurate error estimates for fitting results using 
     `lmfit<http://lmfit.github.io/lmfit-py/>`_.
  *  Analyze lifetime-density using regularized regression.
  *  Analyze 2D spectroscopy data, including CLS-decay, diagonal extraction,
     pump-slice-amplitude spectrum, integration, and Gaussian fitting.

Users
-----

At the moment it is mostly me and other people in my groups. I would be happy
if anyone would like to join the project!

Citiation
---------
If you use this package in your research, please cite the zenodo entry at the top.


License
=======
Standard BSD-License. See the LICENSE file for more details.