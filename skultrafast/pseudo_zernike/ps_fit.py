# %%
from sklearn.linear_model import Ridge
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
from skultrafast.data_io import get_processed_twodim_dataset
from skultrafast.pseudo_zernike import Polybase
%load_ext autoreload
%autoreload 2
# %%

# %%
ds = get_processed_twodim_dataset(True)
# %%
mm = ds.get_minmax(0.3)
xmin = (mm["ProbeMin"] + mm["ProbeMax"]) / 2
xmin = mm["ProbeMin"]
ymin = mm["PumpMin"]
x = ds.probe_wn - xmin
y = ds.pump_wn - ymin
w = 15
x /= w
y /= w


fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})

plt.pcolormesh(x, y, ds.spec2d[3].T, cmap="twilight_r")
ax.add_patch(Circle((0, 0), 1, fill=False))
# %%

# %%


# %%
# Now nbase is 3D array with shape (n, x, y)
# We need to flatten x and y to 1D array
# Then we can use lstsq to get the coefficients

# %%
# Alternatively, since the pseudo-Zernike polynomials are orthogonal, we can
# calculate the coefficients by taking the dot product of the data and the
# pseudo-Zernike polynomials
rmsl = []
for nmax in range(5, 6):
    pb = Polybase(x, y, nmax)
    base = pb.make_base()
    # data /= np.abs(data).max()
    nbase = np.nan_to_num(base)

    c = []
    for t_idx in range(ds.t.size)[:]:
        data = ds.spec2d[t_idx].copy().T
        data[pb.r >= 1] = 0
        data /= np.abs(data).max()
        # coeffs = np.linalg.lstsq(
        #    nbase.reshape(nbase.shape[0], -1).T, data.flatten(),
        # )[0]
        # coeffs = np.sum(nbase.reshape((pb.n+1)**2, -1).T * data.flatten()[:, None], 0)
        mod = Ridge(alpha=1e-1)
        res = mod.fit(nbase.reshape(nbase.shape[0], -1).T, data.flatten())
        # fit = (coeffs[:, None, None] * nbase).sum(axis=0)
        fit = res.predict(nbase.reshape(
            nbase.shape[0], -1).T).reshape(data.shape)
        coeffs = res.coef_
        rms = ((fit - data)**2).mean()
        if t_idx == 0:
            rmsl.append(rms)
        c.append(coeffs)

plt.plot(rmsl, ' o')
# %%
plt.pcolormesh(x, y, fit, cmap="twilight_r")

# %%


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


cs = np.array([cos_sim(c[0], c[i]) for i in range(len(c))])
cd = np.array([cos_sim(c[i], c[-1]) for i in range(len(c))])

plt.plot(ds.t, cs, lw=2, c="r")
plt.plot(ds.t, cd)
pzs = np.cos(np.pi/4)*cd + (1-np.cos(np.pi/4))*cs

plt.plot(ds.t, pzs)
plt.xscale('log')
plt.twinx()
cls_result.plot_cls()

# %%
for i in range(4):
    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"}, figsize=(3, 3))
    plt.pcolormesh(x, y, base[i])
    plt.colorbar()
# %%
plt.plot(c[0])
plt.plot(c[1])
plt.plot(c[2])
plt.plot(c[3])
# %%
cls_result = ds.cls(pu_range=7)
cls_result.plot_cls()
# %%

# %%
