Importing Data
==============
It is very easy to import data into skultrafast, since all what *skultrafast*
requires are numpy arrays. There writing a loader function using Python should
be straight forward.

skultrafast also offer various methods for data pre-processing. Currently, the
package focus is on working with data from *MessPy*, the software for
controlling the experiments in our lab.

A tutorial can be found under :ref:`sphx_glr_auto_examples_tutorial_messpy.py`.


Working with MessPy-files
-------------------------
MessPy saves the data in ``.npz`` file. All data preprocessing and averaging is
now done via the `messpy.MessPyFile` class.

The file format of MessPy is a ``.npz`` file, containing three arrays. The
wavelength, the delay-times in fs and the data. The shape of the data-array is
explained in my PhD-thesis. Loading MessPy is done via the `MessPyFile`-class.
The constructor takes all necessary information to average the data down into
datasets.

This is done via `MessPyFile.average_scans`, which either returns a
``TimeResSpec`` or dict of ``TimeResSpec``'s. For data recorded after 2017, the
following recipes should work.

How to *load data from Vis-pump Vis-Probe data, not polarisation resolved?*::

    mf = MessPyFile(file_name, valid_channel=0, is_pol_resolved=False) data_set
    = mf.average_scans()

How to *load data from Vis-pump Vis-Probe data, polarisation resolved?* For that
we need to know the polarisation of the first scan. The code assumes, that the
polarization of is switched every scan::

    mf = MessPyFile(file_name, valid_channel=0, is_pol_resolve=True,
                    pol_first_scan='perp') #or 'para'
    data_set_dict = mf.average_scans()

How to *load data from IR-Probe setup?* Same as above, but with
``valid_channel=1``, since the zeroth channel contains the unreferenced data.




