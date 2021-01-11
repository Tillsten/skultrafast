"""Here we mostly test if it works at all."""
from skultrafast.dataset import TimeResSpec, PolTRSpec
from skultrafast.data_io import load_example
import numpy as np
from numpy.testing import assert_almost_equal

wl, t, data = load_example()

def test_integrate():
    ds = TimeResSpec(wl, t, data)
    ds.wn_i(15000, 20000)

def test_methods():
    ds = TimeResSpec(wl, t, data)
    bds = ds.bin_freqs(300)
    ds2 = TimeResSpec(1e7/wl, t, data, freq_unit='cm', disp_freq_unit='cm')
    bds2 = ds2.bin_freqs(50)
    assert(np.all(np.isfinite(bds2.data)))

    assert (len(bds.wavelengths) == 300)
    nds = ds.cut_freqs([(400, 600)])
    assert (np.all(nds.wavelengths > 600))
    nds = ds.cut_freq(400, 600)
    assert (np.all(nds.wavelengths > 600))
    nds = ds.cut_times([(-100, 1)])
    assert (np.all(nds.t > .99))
    nds = ds.cut_time(-100, 1)
    assert (np.all(nds.t > .99))
    nds = ds.bin_times(5)
    assert (nds.t.size == np.ceil(ds.t.size / 5))
    ds.mask_freqs([(400, 600)])
    assert (np.all(ds.data.mask[:, ds.wl_idx(550)]))
    ds2 = ds.scale_and_shift(2, t_shift=1, wl_shift=10)
    assert_almost_equal(2*ds.data, ds2.data)
    assert_almost_equal(ds.t+1, ds2.t)
    assert_almost_equal(ds.wavelengths+10, ds2.wavelengths)
    assert_almost_equal(1e7/ds2.wavelengths, ds2.wavenumbers)


def test_est_disp():
    ds = TimeResSpec(wl, t, data)
    ds.auto_plot = False
    for s in ['abs', 'diff', 'gauss_diff', 'max']:
        ds.estimate_dispersion(heuristic=s)


def test_fitter():
    ds = TimeResSpec(wl, t, data)
    x0 = [0.1, 0.1, 1, 1000]
    out = ds.fit_exp(x0)


def test_merge():
    ds = TimeResSpec(wl, t, data)
    nds = ds.merge_nearby_channels(10)
    assert (nds.wavelengths.size < ds.wavelengths.size)


def test_pol_tr():
    ds = TimeResSpec(wl, t, data)
    ds2 = TimeResSpec(wl, t, data)
    ps = PolTRSpec(para=ds, perp=ds2)
    out = ps.bin_freqs(10)
    assert (out.para.wavenumbers.size == 10)
    assert (out.perp.wavenumbers.size == 10)
    assert_almost_equal(out.perp.data, out.para.data)
    ps.subtract_background()
    ps.mask_freqs([(400, 550)])
    print(ps.para.data.mask, ps.para.data.mask[1, ps.para.wl_idx(520)])

    assert (ps.para.data.mask[1, ps.para.wl_idx(520)])
    out = ps.cut_freqs([(400, 550)])
    assert (np.all(out.para.wavelengths >= 550))
    assert (np.all(out.perp.wavelengths >= 550))
    ps.bin_times(6)
    ps.scale_and_shift(1, 0.5)
    ps.copy()


def test_plot():
    ds = TimeResSpec(wl, t, data)
    ds = ds.bin_freqs(50)
    ds.plot.trans([550])
    ds.plot.spec([2, 10])
    ds.plot.trans_integrals((1e7 / 550, 1e7 / 600))
    ds.plot.trans_integrals((1e7 / 600, 1e7 / 500))
    ds.plot.trans([550], norm=1)
    ds.plot.trans([550], norm=1, marker='o')
    ds.plot.map(plot_con=0)
    ds.plot.svd()


def test_pol_plot():
    ds = TimeResSpec(wl, t, data)
    ds = ds.bin_freqs(50)
    ds = PolTRSpec(para=ds, perp=ds)
    ds = ds.bin_freqs(50)
    ds.plot.trans([550])
    ds.plot.spec([2, 10])
    ds.plot.trans([550], norm=1)
    ds.plot.trans([550], norm=1, marker='o')
