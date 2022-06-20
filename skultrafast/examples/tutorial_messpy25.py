"""
Messpy 2.5 Tutorial
===================
This example demonstrates how to load MessPy 2.5 files.
"""
# %%
import numpy as np
import pymesh
import mcubes
import h5py
from skultrafast.twoD_dataset import ContourOptions
from skultrafast.messpy import Messpy25File
from matplotlib import pyplot as plt
from skultrafast import plot_helpers
%load_ext autoreload
%autoreload 2

# %%


# %%
plot_helpers.enable_style()

f = h5py.File(
    r"C:\Users\TillStensitzki\Nextcloud\AG Mueller-Werkmeister\2DIR\tmp2d\22-02-09 13_16 1 M NaSCN in H2O 10 mu.messpy")
# %%
mp = Messpy25File(f)

two_d = mp.get_means()
mp.make_model_fitfiles(r'C:/Users/TillStensitzki/Desktop/test',
                       'test', probe_filter=2, bg_correct=(5, 15))
two_d = mp.make_two_d(probe_filter=1, bg_correct=(10, 10))['iso']

two_d = two_d.select_range((2000, 2155),  (1985, 2155)).select_t_range(0, 6)
co = ContourOptions(levels=16)

two_d.plot.contour(
    0.2, 0.4, 0.8, 1.6, 3, 6, contour_ops=co, ax_size=1.9, average=1, fig_kws={'dpi': 150})
plt.savefig('contour.svg')
ds = two_d
# %%
y_cls, x_cls, lin_fit = ds.single_cls(2, pr_range=20, pu_range=15, method='fit')

# Plot the result
_, ax = ds.plot.contour(2)
ax = ax[0]

# First plot the maxima
ax.plot(x_cls, y_cls, color='yellow', marker='o', markersize=3, lw=0)

# Plot the resulting fit. Since the slope is a function of the pump frequencies,
# we have to use y-values as x-coordinatetes for the slope.
ax.plot(y_cls * lin_fit.slope + lin_fit.intercept, y_cls, color='w')
# %%
cls_result = two_d.cls(pr_range=15, pu_range=15)
cls_result.plot_cls()
two_d.t.shape
# %%
plt.plot(two_d.probe_wn, two_d.spec2d[1, :, two_d.pump_idx(2228)])

# %%

idx1 = two_d.spec2d[0, ...].max(1).argmax()
idx2 = (two_d.spec2d)[0, ...].min(1).argmin()
two_d.probe_wn[[idx1, idx2]]
# %%
tf = two_d.apply_filter('gaussian', (2, 2, 2))

d = tf.spec2d[4:8, :, :].mean(0)
# %%
mcubes.marching_cubes(d, 10)
# %%
# %%

np.save("spec.npy", d)
# %%
m = abs(d).max()
levels = np.linspace(-m, m, 12)
fig = plt.figure(dpi=200)
ax = plt.axes([0, 0, 1, 1])
ax.contourf(d.T, cmap="gray", levels=levels)
plt.savefig('contour.png', dpi=600)
# %%

# %%
