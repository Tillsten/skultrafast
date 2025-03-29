import dataclasses as dc
import matplotlib.pyplot as plt
import numpy as np

import numba as nb

from math import factorial

from functools import lru_cache

@lru_cache(maxsize=None)
def factor(n, l, k):
    nummerator = (-1) ** k * factorial(2*n + 1 - k)
    denummerator = (
        factorial(k)
        * factorial(n + abs(l) + 1 - k)
        * factorial(n - abs(l)  - k)
    )
    return nummerator // denummerator


@nb.njit(fastmath=True)
def _build_cache(r, om, n):
    poly = r[None, :, :] ** np.arange(n + 1)[:, None, None]
    ex = np.exp(1j * np.arange(n + 1)[:, None, None] * om[None, :, :])
    return poly, ex

def S(n, l, rho):
    fac = np.math.factorial
    out = np.zeros_like(rho)
    for k in range((n - abs(l)) + 1):
        num = (-1) ** k * fac(2 * n - k)
        denum = fac(k) * fac(n + abs(l) + 1 - k) * fac(n - abs(l) - k)
        out += (num / denum)* rho ** (n - k)
    return out

def W(n, m, rho, phi):
    return S(n, m, rho) * np.exp(1j * m * phi)

@dc.dataclass
class Polybase:
    """
    Class for generating pseudo-Zernike polynomials.

    Parameters
    ----------
    x : np.ndarray
        x-axis
    y : np.ndarray
        y-axis
    n : int
        Maximum order of the polynomials


    Attributes
    ----------
    r : np.ndarray
        The radial coordinate
    om : np.ndarray
        The azimuthal coordinate
    polybase : np.ndarray
        The polynomial base

    """
    x: np.ndarray
    y: np.ndarray
    n: int
    r: np.ndarray = dc.field(init=False)
    om: np.ndarray = dc.field(init=False)
    polybase: np.ndarray = dc.field(init=False)
    val_dict: dict[tuple[int, int], int] = dc.field(init=False)

    _polycache: np.ndarray = dc.field(init=False)
    _expcache: np.ndarray = dc.field(init=False)

    def __post_init__(self):
        x, y = np.meshgrid(self.x, self.y)
        self.r = np.sqrt(x**2 + y**2)
        self.r = np.where(self.r < 1, self.r, np.nan)
        self.om = np.arctan2(y, x)
        self._polycache, self._expcache = _build_cache(self.r, self.om, self.n)
        self._Rcache = dict()
        self.val_dict = dict()

    def make_base(self):
        l = []
        l.append(np.where(self.r < 1, 1.0, np.nan))
        self.val_dict[(0, 0)] = 0
        for n in range(1, self.n + 1):
            for m in range(0, n + 1):
                arr = self._V(n, m)
                l.append(arr.real)
                self.val_dict[(n, m)] = len(l) - 1
                if m > 0:
                    l.append(arr.imag)
                    self.val_dict[(n, -m)] = len(l) - 1
        return np.array(l)

    def _R(self, n, m):
        R = np.zeros(self.r.shape)
        hit, not_hit = 0, 0
        for s in range((n - abs(m)) + 1):
            if (n, m, s) not in self._Rcache:
                self._Rcache[(n, m, s)] = factor(n, m, s) * \
                    self._polycache[n - s]
                not_hit += 1
            else:
                hit += 1
            R += self._Rcache[(n, m, s)]
        return R

    def _V(self, n, m):
        assert n >= 0
        assert n - abs(m) >= 0
        a = self._R(n, m) * self._expcache[m]
        return a

    def plot_pz(self, n, m, mode="real", offset=(0, 0)):
        plt.gca().set_aspect("equal")
        if mode == "real":
            plt.pcolormesh(
                self.x + offset[0], self.y + offset[1], np.real(self._V(n, m))
            )
        elif mode == "imag":
            plt.pcolormesh(
                self.x + offset[0], self.y + offset[1], np.imag(self._V(n, m))
            )

