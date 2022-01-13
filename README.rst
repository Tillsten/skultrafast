
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
Skultrafast stands for scikit.ultrafast and is an python package which aims
to include everything needed to analyze data from time-resolved spectroscopy
experiments in the femtosecond domain. Its current features are listed down
below.

The latest version of the software is available on `github <https://github
.com/Tillsten/skultrafast>`_. A build of the documentation can be found at
`Read the docs <https://skultrafast.readthedocs.io/en/latest/>`_. The
documentation includes `Installtion notes <https://skultrafast.readthedocs.io/en/latest/install.html>`_.

The package was created and is maintained by *Till Stensitzki*. Most coding was
done while being employed in the `Heyne group <http://www.physik.fu-berlin
.de/einrichtungen/ag/ag-heyne/>`_ and was therefore founded by the DFG via `SFB
1078 <www.sfb1078.de/>`_ and `SFB 1114 <www.sfb1114.de/>`_. Recent additions
were added while being part of the `Ultrafast Structual Dynamics
<https://www.uni-potsdam.de/usd>`_.



Aims of the project
-------------------
I like to include any kind of algorithm or data structure which comes up in
ultrafast physics. I am also open to add a graphical interface to the
package, but as experience shows, a GUI brings in a lot of maintenance
burden. Hence, the first target is a interactive data-explorer for the
jupyter notebook.


Features
--------
The current releases centers around working with time-resolved spectra:

* Publication ready plots with few lines.
* Global fitting of transient Data: DAS, SAS and compartment modelling.
* Support for polarization resovled datasets.
* Automatic dispersion correction of chriped spectra.
* Easy data processing: Selection, Filtering, Recalibration
* Modern error estimates of the fitting results via
  `lmfit <http://lmfit.github.io/lmfit-py/>`_.
* Lifetime-density analyses using regularization regression.
* 2D spectroscopy: Centerline-slope decay, digonal extraction, pump-slice-amplidude
  spectrum, integration.

This package also tries its best to follow modern software practices. This
includes version control using *git*, continues integration testing via
github action and a decent documentation hosted on `Read the docs`.

Users
-----
At the moment it is mostly me and other people in my group. I would be happy
if anyone would like to join the project!


License
=======
Standard BSD-License. See the LICENSE file for