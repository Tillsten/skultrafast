from skultrafast.unit_conversions import (cm2eV, eV2cm, cm2fs, fs2cm, cm2nm, nm2cm,
                                          cm2THz, THz2cm, cm2kcal, kcal2cm)
import numpy as np


def test_handle_np():
    x1 = np.arange(2, 30, dtype='float')
    x0 = 100.
    for x in x0, x1:
        np.testing.assert_allclose(eV2cm(cm2eV(x)), x)
        np.testing.assert_allclose(cm2fs(fs2cm(x)), x)
        np.testing.assert_allclose(cm2THz(THz2cm(x)), x)
        np.testing.assert_allclose(cm2nm(nm2cm(x)), x)
        np.testing.assert_allclose(cm2kcal(kcal2cm(x)), x)