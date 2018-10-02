Importing and Exporting Data
============================

skultrafast also offer various methods for data pre-processing. Currently,
the package focus is on working with data from *MessPy*, the software for
controlling the experiments in our lab.


Working with MessPy-files
-------------------------
MessPy saves the data in ``.npz`` file. All data preprocessing and averaging
is now done via the `messpy.MessPyFile` class.

