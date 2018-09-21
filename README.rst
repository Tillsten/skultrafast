skultrafast
===========
WARNING! THE PROJECT IS A RESTRUCTURING PHASE AND NOT VERY USABLE FOR NEW USERS!

What is skultrafast?
====================

Skultrafast stands for scikit.ultrafast and is an python package which aims
to include everything needed to analyze data from time-resolved spectroscopy
experiments in the femtosecond domain.

The package was created and is maintained by *Till Stensitzki*. All work was
done while employed in the `Heyne group <http://www.physik.fu-berlin
.de/einrichtungen/ag/ag-heyne/>` and was therefore founded by the DFG via SFB
1078 and SFB 1114.

Aims of the project
-------------------
I would include any kind of algorithm or data structure which comes up in
ultrafast physics. Also, data exploration should be made easy (a.k.a. has a
gui for these parts).

Features
--------
The current releases center around time-resolved spectra.
    * Publication ready plots with few lines.
    * Automatic dispersion correction.
    * Very fast exponential fitting, which can make use of your GPU.
    * Lifetime-density analyses using regularization regression.

Users
-----
At the moment it is mostly me and other people in my group. I would be happy
if anyone would like to join the project!

Basic Usage
===========

Dataset
-------
New users should use the `DataSet` class, which provides methods to explore,
fit, preprocess and visualize time-resolved spectra. The class makes uses of
older parts of the package. With time, I plan to deprecate older parts of the
package and move all functionality to classes like `DataSet`.

Software prerequisites
=======================
Python 3 only. Requires numpy, scipy, lmfit, matplotlib, sklearn, statsmodels.

For a maximum of speed when fitting exponenitals `pyopencl` should be
installed.


License
-------

Standard BSD-Licence.

