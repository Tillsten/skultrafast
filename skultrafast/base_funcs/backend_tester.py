import numpy as np
import numba as nb
from scipy.special import erfc

from math import erfc
import time
from collections import namedtuple
import matplotlib.pyplot as plt

@nb.vectorize(fastmath=True)
def my_erfc(x):
    return erfc(x)


BenchResult = namedtuple('BenchResult', 'all mean std')


def benchmark(func, ta_shape=(1000, 400), N=100):
    t_array = np.subtract.outer(np.linspace(-1, 50, ta_shape[0]),
                                np.linspace(-3, 3, ta_shape[1]))
    w = 0.1
    taus = np.array([0.1, 2, 10, 1000])
    #Run once for jit
    func(t_array, w, 0., taus)
    out = []
    for i in range(N):
        t = time.time()
        func(t_array, w, 0., taus)
        out.append(time.time() - t)
    out = np.array(out)
    mean = out.mean()
    std = out.std()
    return BenchResult(out, mean, std)


def fast_erfc(x):
    """
    Calculates the erfc near zero faster than
    the libary function, but has a bigger error, which
    is not a problem for us.

    Parameters
    ----------
    x: float
        The array

    Returns
    -------
    ret: float
        The erfc of x.
    """
    a1 = 0.278393
    a2 = 0.230389
    a3 = 0.000972
    a4 = 0.078108
    smaller = x < 0
    if smaller:
        x = x * -1.
    bot = 1 + a1 * x + a2 * x * x + a3 * x * x * x + a4 * x * x * x * x
    ret = 1. / (bot * bot * bot * bot)

    if smaller:
        ret = -ret + 2.

    return ret


my_erfc = nb.vectorize(fast_erfc)


def _fold_exp(tt, w, tz, tau):
    """
    Returns the values of the folded exponentials for given parameters.

    Parameters
    ----------
    tt:  ndarray(N)
        Array containing the time-coordinates
    w:  float
        The assumed width/sq2
    tz: float
        The assumed time zero.
    tau: ndarray(M)
        The M-decay rates.

    Returns
    -------
    y: ndarray(N,M)
       Folded exponential for given taus.

    """
    ws = w
    k = 1 / tau
    k = k.reshape(tau.size, 1, 1)

    t = (tt + tz).T.reshape(1, tt.shape[1], tt.shape[0])
    y = np.zeros_like(t)
    idx = (np.abs(t/w) < 3)
    y[idx] = np.exp(k * (ws * ws * k / (4.0) - t[idx])) * 0.5 * my_erfc(-t[idx] / ws + ws * k / (2.0))
    #y = np.exp(k * (ws * ws * k / (4.0) - t))
    #y *= 0.5 * my_erfc(-t / ws + ws * k / (2.0))
    return y.T


jit1 = nb.njit(_fold_exp, parallel=True, fastmath=True)
jit2 = nb.njit(_fold_exp, parallel=False, fastmath=True)
jit3 = nb.njit(_fold_exp, parallel=True, fastmath=False)
jit4 = nb.njit(_fold_exp, parallel=False, fastmath=False)

from base_functions_numba import _fold_exp as jit5

plt.figure()
for i, j in enumerate([jit1, jit2, jit3, jit4, jit5]):
    for N in np.geomspace(10, 1000, 5):
        res_jit = benchmark(j, ta_shape=(300, int(N)), N=30)
        plt.plot(np.ones_like(res_jit.all) * N,
                 res_jit.all,
                 'x',
                 c='C' + str(i))

plt.show()
