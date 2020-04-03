"""
Measuring the system response in the mid IR
===========================================

In the mid-IR the system response is measured by monitoring the transmittance of the
probe light through a thin semi-conductor. skultrafast has an helper function to
analyze such a signal.
"""
# %%
from skultrafast import messpy, data_io

fname = data_io.get_example_path('sys_response')
tz_result = messpy.get_t0(fname, display_result=False,
                          t_range=(-2, 0.3),
                          no_slope=False)

# %%
# Newer version of lmfit have a html representation which is used by ipython, e.g.
# in the notebook. Hence the line below will display the fit results.

tz_result.fit_result

