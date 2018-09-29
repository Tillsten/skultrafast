from skultrafast.dataset import DataSet
from skultrafast.data_io import  load_example
import numpy as np

wl, t, data = load_example()

def test_methods():
    ds = DataSet(wl, t, data)
    bds = ds.bin_freqs(300)
    assert(len(bds.wavelengths) == 300)
    nds = ds.cut_freqs([(400, 600)])
    assert(np.all(nds.wavelengths > 600))


