import matplotlib.pyplot as plt

import numpy as np
from skultrafast import unit_conversions
from lmfit import model
import lmfit
import inspect
import sympy


def cosd(deg):
    return sympy.cos(deg / 180 * sympy.pi)


def angle_to_dichro(deg):
    return (1 + 2 * cosd(deg)**2) / (2 - cosd(deg)**2)


def make_pol(func):
    def wrapper(*args):
        angle = args[-1]
        print(angle)
        perp = func(*args[:-1])
        para = perp * angle_to_dichro(angle)
        return perp, para

    return wrapper


@make_pol
def lorentz(wl, t, A, Ac, xc, w, tau):
    x = (wl-xc) / w
    return (A * sympy.exp(-t / tau) + Ac) * 1 / (1 + x**2)


@make_pol
def gauss(wl, t, A, Ac, xc, w, tau):
    x = (wl-xc) / w
    return (A * sympy.exp(-t / tau) + Ac) * np.exp(-.5 * x**2)


@make_pol
def gauss_const(wl, t, A, xc, w):
    x = (wl-xc) / w
    return A * np.exp(-.5 * x**2)


@make_pol
def lorentz_const(wl, t, A, xc, w):
    x = (wl-xc) / w
    return A / (1 + x**2)


class ModelBuilder:
    def __init__(self, wl: np.ndarray, t: np.ndarray):
        self.funcs = []
        self.args = []
        self.values = {}
        self.n = 0
        self.t, self.wl = sympy.symbols('t wl')
        self.t_arr = t
        self.wl_arr = wl
        self.coords = [self.wl, self.t]

    def add_decaying(self,
                     A: float,
                     Ac: float,
                     xc: float,
                     w: float,
                     tau: float,
                     angle: float,
                     peak_type: str = 'lor') -> int:
        names = "A Ac xc w tau angle"
        all_sym = [self.wl, self.t]
        for name in names.split(' '):
            sym = '%s_%d' % (name, self.n)
            all_sym.append(sympy.Symbol(sym))
            self.values[sym] = locals()[name]

        if peak_type == 'lor':
            self.funcs.append(lorentz(*all_sym))
        elif peak_type == 'gauss':
            self.funcs.append(gauss(*all_sym))
        self.n += 1
        self.args += all_sym[2:]
        return len(self.funcs)

    def add_constant(self, A, xc, w, angle, peak_type='lor'):  #
        names = 'A xc w angle'
        all_sym = [self.wl, self.t]
        for name in names.split(' '):
            sym = '%s_%d' % (name, self.n)
            all_sym.append(sympy.Symbol(sym))
            self.values[sym] = locals()[name]

        if peak_type == 'lor':
            self.funcs.append(lorentz_const(*all_sym))
        elif peak_type == 'gauss':
            self.funcs.append(gauss_const(*all_sym))
        self.n += 1
        self.args += all_sym[2:]
        return len(self.funcs)

    def make_model(self):
        para = []
        perp = []
        for i in range(0, self.n):
            perp.append(self.funcs[i][0])
            para.append(self.funcs[i][1])
        all_para = sum(para)
        all_perp = sum(perp)
        return all_para, all_perp

    def make_params(self):
        pa, pe = self.make_model()
        expr = sympy.Tuple(pa, pe)

        free = expr.free_symbols
        func = sympy.lambdify(list(free), expr)

        mod = lmfit.Model(func, ['wl', 't'])
        params = mod.make_params()
        for pn in params:
            p = params[pn]
            p.value = self.values[pn]
            if pn.startswith('angle'):
                p.min = 0
                p.max = 90
            elif pn.startswith('w'):
                p.min = 2
        y = mod.eval(wl=self.wl_arr[:, None], t=self.t_arr, **params)

        return params, mod

    def plot_peaks(self, params=None):
        if params is None:
            params, mod = self.make_params()

        para = sympy.Tuple(*(i[1] for i in self.funcs))
        perp = sympy.Tuple(*(i[0] for i in self.funcs))

        pa_func = sympy.lambdify(self.coords + list(params.keys()), para)
        pe_func = sympy.lambdify(self.coords + list(params.keys()), perp)

        pa = pa_func(t=self.t_arr, wl=self.wl_arr[:, None], **params)
        pe = pe_func(t=self.t_arr, wl=self.wl_arr[:, None], **params)
        for a, b in zip(pa, pe):
            l1, = plt.plot(a[:, 0], lw=2)
            plt.plot(b[:, 0], c=l1.get_color())
        y = mod.eval(wl=self.wl_arr[:, None], t=self.t_arr, **params)
