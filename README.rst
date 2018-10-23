skultrafast
***********
.. image:: https://readthedocs.org/projects/skultrafast/badge/?version=latest
    :target: https://skultrafast.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/Tillsten/skultrafast.svg?branch=master
    :target: https://travis-ci.org/Tillsten/skultrafast

**WARNING! THE PROJECT IS A RESTRUCTURING PHASE AND NOT VERY USABLE FOR NEW
USERS!**

What is skultrafast?
====================
Skultrafast stands for scikit.ultrafast and is an python package which aims
to include everything needed to analyze data from time-resolved spectroscopy
experiments in the femtosecond domain. Its current features are listed further
below.

The latest version of the software is available on `github <https://github
.com/Tillsten/skultrafast>`_. A build of the documentation can be found at
`Read the docs <https://skultrafast.readthedocs.io/en/latest/>`_.

The package was created and is maintained by *Till Stensitzki*. All coding was
done while being employed in the `Heyne group <http://www.physik.fu-berlin
.de/einrichtungen/ag/ag-heyne/>`_ and was therefore founded by the DFG via
`SFB 1078 <www.sfb1078.de/>`_ and `SFB 1114 <www.sfb1114.de/>`_.

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
* Automatic dispersion correction.
* Easy data processing.
* Very fast exponential fitting, which can make use of your GPU.
* Modern error estimates of the fitting results via
  `lmfit <http://lmfit.github.io/lmfit-py/>`_.
* Lifetime-density analyses using regularization regression.

This package also tries its best to follow modern software practices. This
includes version control using *git*, continues integration testing via
travisCI and decent documentation.

Users
-----
At the moment it is mostly me and other people in my group. I would be happy
if anyone would like to join the project!


License
=======
Standard BSD-License. See the LICENSE file for details.

