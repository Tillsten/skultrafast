from skultrafast.utils import pfid_r4, pfid_r6, simulate_binning
import numpy as np
import pytest

def test_pfid():
    t = np.linspace(0, 10, 100)
    fre = np.linspace(900, 1200, 64)
    y1 = pfid_r4(t, fre, [1000, 1100], [2, 1])
    y2 = pfid_r6(t, fre, [1000], [1015], [2])


@pytest.mark.skip("Not implemented yet")
def test_simulate_binning():
    wl = np.linspace(0, 2*np.pi, 4)

    def func(*, wl):
        return np.sin(wl)

    func_bin = simulate_binning(func, fac=20)
    res = func(wl=wl)
    binned_res = func_bin(wl=wl)
    assert res.shape == binned_res.shape
    many = np.linspace(0, 2*np.pi, 10_0000)
    precise_sum = np.sin(many)




