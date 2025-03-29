# %%
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import Ridge
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
from skultrafast.data_io import get_processed_twodim_dataset
from skultrafast.pseudo_zernike import Polybase

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
w = 12
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
for nmax in range(5, 15):
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
        mod = Ridge(alpha=1e-3, fit_intercept=False)
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
cls_result = ds.cls(pu_range=7)
# %%


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


cs = np.array([cos_sim(c[0], c[i]) for i in range(len(c))])
cd = 1-np.array([cos_sim(c[i], c[-1]) for i in range(len(c))])

# plt.plot(ds.t, cs, lw=2, c="r")
# plt.plot(ds.t, cd)
pzs = np.cos(np.pi/4)*cs + (1-np.cos(np.pi/4))*cd

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

cls_result.plot_cls()
# %%

# %%


# %%
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

ptest = Polybase(x, y, 5)
nbase = ptest.make_base()
ptest
inv_val_dict = {v: k for k, v in ptest.val_dict.items()}
nbase = np.nan_to_num(nbase)
for i, b in enumerate(nbase):
    for j, b2 in enumerate(nbase):
        if i == j or i == 0 or j == 0:
            pass
        else:
            val = np.dot(b.flatten(), b2.flatten() )
            if abs(val) > 1e-3:
                print( "%.1f"%np.dot(b.flatten(), b2.flatten() ), inv_val_dict[i], inv_val_dict[j])
# %%
plt.imshow(nbase[ptest.val_dict[(1, 0)]] )
plt.figure()
plt.imshow(nbase[ptest.val_dict[(3, 0)]] )
# %%

def zernike_radial_polynomials(n, r):
    if np.any((r > 1) | (r < 0) | (n < 0)):
        raise ValueError('r must be between 0 and 1, and n must be non-negative.')

    if n == 0:
        return np.ones(len(r))

    R = np.zeros((n + 1, len(r)))
    r_sqrt = np.sqrt(r)
    r0 = r_sqrt ** (2 * n + 1) != 0

    if np.any(r0):
        R[0, r0] = 1

    if np.any(~r0):
        r_sqrt = r_sqrt[~r0]
        R1 = zernike_radial_polynomials(n - 1, r_sqrt)
        m = np.arange(2, 2 * n + 2, 2)
        R1 = R1[m, :]
        R1 = R1 / r_sqrt[:, None]
        R[:, ~r0] = R1

    return R

zernike_radial_polynomials(3, np.linspace(0, 1, 2))
# %%
