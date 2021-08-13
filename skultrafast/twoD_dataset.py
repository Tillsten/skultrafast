from pathlib import Path
from sys import prefix
from typing import Dict, Iterable, List, Optional, Tuple, Union

import attr
import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.ndimage import  map_coordinates
from scipy.interpolate import RegularGridInterpolator
import proplot

from skultrafast import dv, plot_helpers
from skultrafast.dataset import TimeResSpec


def inbetween(a, lower, upper):
    return np.logical_and(a >= lower, a <= upper)


@attr.s(auto_attribs=True)
class CLSResult:
    wt: List[float]
    slopes: List[float]
    slope_errors: Optional[List[float]] = None
    lines: Optional[np.ndarray] = None
    exp_fit_result_: Optional[lmfit.model.ModelResult] = None

    def exp_fit(self, start_taus: List[float], use_const=True, use_weights=True):
        mod = lmfit.models.ConstantModel()
        vals = {}

        for i, v in enumerate(start_taus):
            prefix = 'abcdefg'[i] + '_'
            mod += lmfit.models.ExponentialModel(prefix=prefix)
            vals[prefix + 'decay'] = v
            vals[prefix + 'amplitude'] = 0.5 / len(start_taus)

        for p in mod.param_names:
            if p.endswith('decay'):
                mod.set_param_hint(p, min=0)
            if p[:3] == 'amp':
                mod.set_param_hint(p, min=0)
        mod.set_param_hint('c', min=0)

        if self.slope_errors is not None and use_weights:
            weights = 1 / np.array(self.slope_errors)
        else:
            weights = None
        res = mod.fit(self.slopes,
                      weights=weights,
                      x=self.wt,
                      c=max(np.min(self.slopes), 0),
                      **vals)
        self.exp_fit_result_ = res
        return res


    def plot_cls(self, ax=None, model_style: Dict=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ec = ax.errorbar(self.wt, self.slopes, self.slope_errors, **kwargs)
        plot_helpers.lbl_trans(ax=ax)
        ax.set(xlabel=plot_helpers.time_label, ylabel='Slope')
        m_line = None
        if self.exp_fit_result_:
            xu = np.linspace(min(self.wt), max(self.wt), 300)
            yu = self.exp_fit_result_.eval(x=xu)
            style=dict(c='k', zorder=1.8)
            if model_style: style.update(model_style)
            ax.plot(xu, yu, color='k', zorder=1.8)
        return ec, m_line


@attr.s(auto_attribs=True)
class TwoDimPlotter:
    ds: 'TwoDim' = attr.ib()

    def contour(self, *times, region=None, ax=None, subplots_kws={}, aspect=None):
        ds = self.ds
        idx = [ds.t_idx(i) for i in times]
        if ax is None:
            if aspect is None:
                aspect = ds.probe_wn.ptp() / ds.pump_wn.ptp()
            fig, ax = proplot.subplots(nrows=len(idx), **subplots_kws,
                             aspect=aspect)

        m = abs(ds.spec2d).max()
        for i, k in enumerate(idx):
            c = ax[i].contourf(ds.probe_wn,
                               ds.pump_wn,
                               ds.spec2d[k].T,
                               N=np.linspace(-m, m, 21),
                               cmap='Div',
                               symmetric=True,
                               linewidths=[0.7],
                               linestyles=['-'])
            start = max(ds.probe_wn.min(), ds.pump_wn.min())
            ax[i].axline((start, start), slope=1, c='k', lw=0.5)
            if ds.t[k] < 1:
                title = '%.0f fs' % (ds.t[k] * 1000)
            else:
                title = '%.1f ps' % (ds.t[k])
            ax[i].format(title=title, titleloc='ul', titleweight='bold')

        ax.format(xlabel='Probe Freq. / cm$^{-1}$',
                  ylabel='Pump Freq. / cm$^{-1}$',
                  xlim=region,
                  ylim=region)
        return c, ax

    def movie_contour(self, fname, subplots_kw={}):
        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots()

        frames = self.ds.t
        ani = FuncAnimation(fig=fig, func=self.contour, frames=frames)
        ani.save(fname)

    def plot_cls(self):
        pass

    def pump_slice_amps(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.ds.pump_wn, self.ds.pump_slice_amp())
        plot_helpers.ir_mode()
        ax.set(xlabel=plot_helpers.freq_label, ylabel='Slice Amp. [mOD]')


@attr.s(auto_attribs=True)
class TwoDim:
    t: np.ndarray
    "Array of the waiting times"
    pump_wn: np.ndarray
    "Array with the pump-wavenumbers"
    probe_wn: np.ndarray
    "Array with the probe-wavenumbers"
    spec2d: np.ndarray
    "Array with the data"
    info: Dict = {}
    "Meta Info"
    cls_result_: Optional[CLSResult] = None
    "Contains the data from a CLS analysis"
    plot: 'TwoDimPlotter' = attr.Factory(TwoDimPlotter, True) #typing: Ignore
    "Plot object offering plotting methods"
    interpolator_: Optional[RegularGridInterpolator] = None #typing: Ignore

    def _make_int(self, ):
        RegularGridInterpolator((self.t, self.probe_wn, self.pump_wn), self.spec2d, bounds_error=False)

    def __attrs_post_init__(self):
        n, m, k = self.t.size, self.probe_wn.size, self.pump_wn.size
        if self.spec2d.shape != (n, m, k):
            raise ValueError("Data shape not equal to t, wn_probe, wn_pump shape"
                             f"{self.spec2d.shape} != {n, m, k}")

        self.spec2d = self.spec2d.copy()
        self.probe_wn = self.probe_wn.copy()
        self.pump_wn = self.pump_wn.copy()

        i1 = np.argsort(self.pump_wn)
        self.pump_wn = self.pump_wn[i1]
        i2 = np.argsort(self.probe_wn)
        self.probe_wn = self.probe_wn[i2]

    def copy(self) -> 'TwoDim':
        cpy = attr.evolve(self)
        cpy.plot = TwoDimPlotter(cpy) #typing: ignore
        return cpy

    def t_idx(self, t: Union[float, Iterable[float]]) -> int:
        "Return nearest idx to nearest time value"
        return dv.fi(self.t, t)

    def probe_idx(self, wn: Union[float, Iterable[float]]) -> int:
        "Return nearest idx to nearest probe_wn value"
        return dv.fi(self.probe_wn, wn)

    def pump_idx(self, wn: Union[float, Iterable[float]]) -> int:
        "Return nearest idx to nearest pump_wn value"
        return dv.fi(self.pump_wn, wn)

    def select_range(self, pump_range, probe_range, invert=False) -> 'TwoDim':
        """
        Return a dataset containing only the selected region.
        """
        pu_idx = inbetween(self.pump_wn, min(pump_range), max(pump_range))
        pr_idx = inbetween(self.probe_wn, min(probe_range), max(probe_range))

        if invert:
            pu_idx = not pu_idx
            pr_idx = not pr_idx

        ds = self.copy()
        ds.spec2d = ds.spec2d[:, pr_idx, :][:, :, pu_idx]
        ds.pump_wn = ds.pump_wn[pu_idx]
        ds.probe_wn = ds.probe_wn[pr_idx]
        return ds

    def intregrate_pump(self, lower: float, upper: float) -> TimeResSpec:
        """
        Calculate and return 1D Time-resolved spectra for given range.

        Parameters
        ----------
        lower : float
            Lower pump wl
        upper : float
            upper pump wl

        Returns
        -------
        TimeResSpec
            The corresponding 1D Dataset
        """
        pu_idx = inbetween(self.pump_wn, lower, upper)
        data = np.trapz(self.spec2d[:, :, pu_idx], self.pump_wn[pu_idx], axis=-1)
        return TimeResSpec(self.probe_wn, self.t, data, freq_unit='cm')

    def single_cls(self, t, pr_range=9, pu_range=7, mode='neg'):
        """
        Calculate the CLS for single 2D spectrum.

        Parameters
        ----------
        t : float
            Delay time of the spectrum to analyse
        pr_range : float, optional
            How many wavenumbers away from the maxium to use for
            determining the exact position, by default 9
        pu_range : float, optional
            The range around the pump-maxima used for calculating
            the CLS.
        mode : str, optional
            negative or positive maxium, by default 'neg'

        Returns
        -------
        (x, y, r)
            Return x, y and the regression result r
        """
        pu = self.pump_wn
        pr = self.probe_wn
        spec = self.spec2d[self.t_idx(t), :, :].T
        if mode == 'pos':
            spec = -spec
        pu_max = pu[np.argmin(np.min(spec, 1))]
        pu_idx = (pu < pu_max + pu_range) & (pu > pu_max - pu_range)
        spec = spec[pu_idx, :]
        l = []

        for s in spec:
            m = np.argmin(s)
            pr_max = pr[m]
            i = u_idx = (pr < pr_max + pr_range) & (pr > pr_max - pr_range)
            cen_of_m = np.average(pr[i], weights=s[i])
            l.append(cen_of_m)

        x = pu[pu_idx]
        y = np.array(l)
        r = linregress(x, y)
        return x, y, r

    def cls(self, **cls_args):
        slopes, slope_errs = [], []
        lines = []
        for d in self.t:
            x, y, r = self.single_cls(d, **cls_args)
            slopes.append(r.slope)
            slope_errs.append(r.stderr)
            lines.append(np.column_stack((x, y)))
        res = CLSResult(self.t, slopes=slopes, slope_errors=slope_errs, lines=lines)
        self.cls_result_ = res
        return res

    def extract_line_spec(self, offset):
        if not self.interpolator_:
            self.interpolator_ = self._make_int()



    def pump_slice_amp(self, method='minmax'):
        sla = np.ptp(self.spec2d, 0)