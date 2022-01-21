"""
Messpy 2.5 Tutorial
===================
This example demonstrates how to load MessPy 2.5 files.
"""

# %%

# %%
from matplotlib import pyplot as plt
from skultrafast import plot_helpers
plot_helpers.enable_style()
from skultrafast.messpy import Messpy25File
import h5py

f = h5py.File(r"D:\boxup\AG Mueller-Werkmeister\2DIR\tmp2d\22-01-21 17_27 test_benzonitril.messpy")
mp = Messpy25File(f)

two_d = mp.get_means()
two_d = mp.make_two_d()['para']
two_d = two_d.select_range((2200, 2260), (2180, 2260))
two_d.plot.contour(0.2, 0.4, 0.8, 1.6,  3.2, 6.4)
#
# %%
cls_result = two_d.cls()
cls_result.plot_cls()
# %%
