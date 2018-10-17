"""Here we mostly test if it works at all."""
from skultrafast.dataset import TimeResSpec, PolTRSpec
from skultrafast.data_io import  load_example
import numpy as np

wl, t, data = load_example()

def test_methods():
    ds = TimeResSpec(wl, t, data)
    bds = ds.bin_freqs(300)
    assert(len(bds.wavelengths) == 300)
    nds = ds.cut_freqs([(400, 600)])
    assert(np.all(nds.wavelengths > 600))
    nds = ds.cut_times([(-100, 1)])
    assert(np.all(nds.t > .99))
    nds = ds.bin_times(5)

def test_pol_tr():
    ds = TimeResSpec(wl, t, data)
    ps = PolTRSpec(para=ds, perp=ds)
    ps.bin_freqs(10)
    ps.subtract_background()
    ps.mask_freqs([(500, 550)])
    ps.cut_freqs([(500, 550)])
    ps.bin_times(6)

def test_plot():
    ds = TimeResSpec(wl, t, data)
    ds = ds.bin_freqs(50)
    ds.trans([550])
    ds.spec([2, 10])
    ds.trans([550], norm=1)
    ds.trans([550], norm=1, marker='o')
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
