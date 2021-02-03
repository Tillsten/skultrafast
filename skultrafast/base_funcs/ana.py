# %%
from skultrafast.base_funcs.pytorch_fitter import *
from skultrafast import data_io, dataset
from pathlib import Path
import matplotlib.pyplot as plt
import proplot
ex_dir = Path(__file__).parent.parent / 'examples/data'

dsl, dsbl, dsl2 = [], [], []
for pol in ('iso', 'para', 'perp'):
    cut_wn = 18000
    ds = TimeResSpec.from_txt(ex_dir / f'AGP2 wt (uncorrected).{pol}.txt',
                              disp_freq_unit='cm')
    ds = ds.cut_freq(cut_wn, 1e8)
    ds = ds.cut_time(-100, -.3)
    dsb = TimeResSpec.from_txt(ex_dir / f'Buffer (uncorrected).{pol}.txt',
                               disp_freq_unit='cm')
    dsb = dsb.cut_freq(cut_wn, 1e8)
    dsb = dsb.cut_time(-.15, .2, invert_sel=True)
    dsl.append(ds.copy())
    ds.data[ds.t_idx(-.15):ds.t_idx(-.15)+dsb.t.size] -= dsb.data
    dsl2.append(ds)
    dsbl.append(dsb)

iso, para, perp = dsl
pol = dataset.PolTRSpec(para,perp)
both = para.copy()
both.data = np.hstack((both.data, perp.data))
both.wavenumbers = np.hstack((both.wn, both.wn))

# %%
nt = dsbl[0].estimate_dispersion('abs',(1,), shift_result=0.03)

# %%

fit = FitterTorch(dsbl[0], zero_func=nt.polynomial, disp_poly_deg=3, model_coh=True,
    use_cuda=True)
paras, mini = fit.start_lmfit([0.001, 0.01], [ 0.1], False, False, False)
mini.params['w1'].max = 0.02
res1 = mini.leastsq()

mini.userfcn(res1.params)
res1
# %%
fig, ax = proplot.subplots()
for wn in [11000, 14500, 18000]:
    ax.plot(fit.tt.T[:, iso.wn_idx(wn)], fit.dataset.wn_d(wn), lw=2)
    ax.plot(fit.tt.T[:, iso.wn_idx(wn)], fit.model.cpu().T[:, iso.wn_idx(wn)], lw=1, c='k')
ax.format(xlim=(-.3, .3))
# %%
fig, ax = proplot.subplots()
for wn in [11000, 12000,16000, 18000]:
    ax.plot(dsb.t, fit.dataset.wn_d(wn), lw=2)
    ax.plot(dsb.t, fit.model.cpu().T[:, iso.wn_idx(wn)], lw=1, c='k')
ax.format(xlim=(-.3, .3))




# %%
para, perp, iso = dsl2

both2 = para.copy()
both2.data = np.hstack((both2.data, perp.data))
both2.wavenumbers = np.hstack((both.wn, both.wn))

base = both.data.copy()*0
coh = np.hstack((dsbl[1].data, dsbl[2].data))
base[ds.t_idx(-.15):ds.t_idx(-.15)+dsb.t.size, :] = coh

fit2 = FitterTorch(both2, zero_func=nt.polynomial, disp_poly_deg=3, 
                   model_coh=False, use_cuda=True)#, extra_base=base.T[..., None])

paras2, mini2 = fit2.start_lmfit([0.001, 0.4], [ 0.1,  0.3, 2,  4, 10000],
                                 True,
                                 False,
                                 False,
                                 )
res2 = mini2.leastsq()
mini2.userfcn(res2.params)
res2
# %%
fig, (ax, ax2) = proplot.subplots(nrows=2, axwidth=4, axheight=3, sharex=0)
off = 0
for wn in [11000, 12300, 14500, 18000]:
    ax.plot(fit2.tt.T[:, iso.wn_idx(wn)], fit2.model.cpu().T[:, iso.wn_idx(wn)]+off, 
            lw=1, c='k')
    ax.plot(fit2.tt.T[:, iso.wn_idx(wn)], both2.wn_d(wn)+off, lw=1)
    
    ax.plot(dsbl[1].t - fit2.t_zeros[both.wn_idx(wn)], dsbl[1].wn_d(wn)+off, lw=1)

    ax2.plot(fit2.tt.T[:, iso.wn_idx(wn)], fit2.model.cpu().T[:, iso.wn_idx(wn)]+off, 
            lw=1, c='k')
    ax2.plot(fit2.tt.T[:, iso.wn_idx(wn)], both2.wn_d(wn)+off, lw=1)
    
    ax2.plot(dsbl[1].t - fit2.t_zeros[both.wn_idx(wn)], dsbl[1].wn_d(wn)+off, lw=1)
    off += 15



ax.format(xlim=(-.1, 1.5), xscale='linear')
ax2.format(xlim=(.1, 15), xscale='log')

# %%
from skultrafast import plot_helpers
fig, ax = proplot.subplots(width='4in', height='3in')
proplot.rc['axes.labelsize'] = 'small'
k = iso.wn.size
print(fit.c.shape, iso.wn.size)
end = -3 if fit2.model_coh else None
fit2.c = fit2.c.cpu()
l = ax.plot(both.wavenumbers[:k], fit2.c[:k, :end], lw=1)
l2 = ax.plot(both.wavenumbers[:k], fit2.c[k:, :end], lw=1)
for i, j in zip(l, l2):
    j.set_color(i.get_color())
leg = ['%.2f ps' % v for i, v in res2.params.items() if i[0] == 't']
if fit2.extra_base is not None:
    leg += ['Solvent']
ax.legend(l, leg, ncol=2)
ax.format(xlim=(both.wn.min(), both.wn.max()),
          xlabel=plot_helpers.freq_unit,
          ylabel=plot_helpers.sig_label)
d = (lambda x: 1e7 / x, lambda x: 1e7 / x)
ax[0].dualx(d, label='Wavelength [nm]')
# %%
# %%
ci, ci2 = lmfit.conf_interval(mini2, res2, trace=True)
# %%
ci

# %%
# %%
from matplotlib import pyplot as plt, use
plt.pcolormesh(fit.tt, iso.wn, fit.dev_data, cmap='turbo')
plt.xlim(-0.2, 4)
# %%
polf = pol
polf.plot.trans(12000)
polf.plot.trans(12000)
nt = polf.cut_freq(12000, 14000).iso.estimate_dispersion('diff', (2,))
# %%

# %%
plt.plot(fit.t_zeros)

# %%
plt.imshow(fit2.dev_data-fit2.model)
# %%
plt.plot(fit2.A.cpu()[50, :, 8])
plt.plot(fit2.dev_data.cpu()[50, :])

# %%
