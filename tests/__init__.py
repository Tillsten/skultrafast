
import nose
from skultrafast.fitter import fast_erfc, _fold_exp, _coh_gaussian, full_fast_fit
from numpy.testing import assert_array_almost_equal
import numpy as np

def test_fast_erfc():
    from scipy.special import erfc as erfc_s
    x = np.linspace(3, 3, 200)
    assert_array_almost_equal(erfc_s(x), fast_erfc(x), 4)

def test_fold_exp():
    taus = np.array([1, 20, 30])
    t_array = np.subtract.outer(np.linspace(-1, 50, 300), 
                                np.linspace(3, 3, 400))
    w = 0.1
    
    return full_fast_fit(t_array, w, 0., taus)
    




if __name__ == '__main__':
     import matplotlib.pyplot as plt
     
     a = test_fold_exp()
     
     plt.plot(a[:, 0, :])
     plt.show()
     nose.run()    