from typing import Dict, Iterable, List, Optional, Tuple, Union, Literal

import attr
import lmfit
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

from scipy.stats import linregress, norm
from scipy.ndimage import map_coordinates, uniform_filter1d
from scipy.interpolate import RegularGridInterpolator


from skultrafast import dv, plot_helpers
from skultrafast.dataset import TimeResSpec


def inbetween(a, lower, upper):
    return np.logical_and(a >= lower, a <= upper)


@attr.s(auto_attribs=True)
class CLSResult:
    wt: List[float]
    slopes: List[float]
    intercepts: List[float]
    intercept_errors: Optional[List[float]] = None
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
        # mod.set_param_hint('c', min=-0)
        if use_const:
            c = max(np.min(self.slopes), 0)
        else:
            c = 0
            mod.set_param_hint('c', vary=False)

        if self.slope_errors is not None and use_weights:
            weights = 1 / np.array(self.slope_errors)
        else:
            weights = None
        res = mod.fit(self.slopes,
                      weights=weights,
                      x=self.wt,
                      c=c,
                      **vals)
        self.exp_fit_result_ = res
        return res

    def plot_cls(self, ax=None, model_style: Dict = None, symlog=False, **kwargs):
        if ax is None:
            ax = plt.gca()
        ec = ax.errorbar(self.wt, self.slopes, self.slope_errors, **kwargs)
        if symlog:
            ax.set_xscale('symlog', linthresh=1)
        plot_helpers.lbl_trans(ax=ax, use_symlog=symlog)

        ax.set(xlabel=plot_helpers.time_label, ylabel='Slope')
        m_line = None
        if self.exp_fit_result_:
            xu = np.linspace(min(self.wt), max(self.wt), 300)
            yu = self.exp_fit_result_.eval(x=xu)
            style = dict(c='k', zorder=1.8)
            if model_style:
                style.update(model_style)
            m_line = ax.plot(xu, yu, color='k', zorder=1.8)
        return ec, m_line


@attr.s(auto_attribs=True)
class DiagResult:
    diag: np.ndarray
    antidiag: np.ndarray
    diag_coords: np.ndarray
    antidiag_coords: np.ndarray
    offset: float
    p: float

@attr.s(auto_attribs=True)
class TwoDimPlotter:
    ds: 'TwoDim' = attr.ib()

    def contour(self, *times,  ax=None, ax_size=1.5, subplots_kws={}, aspect=None,
                direction='vertical', scale="firstmax", average=None):
        ds = self.ds
        idx = [ds.t_idx(i) for i in times]
        if ax is None:
            if aspect is None:
                aspect = ds.probe_wn.ptp() / ds.pump_wn.ptp()
            if direction[:1] == 'v':
                nrows = len(idx)
                ncols = 1
            else:
                nrows = 1
                ncols = len(idx)
            if aspect > 1:
                ax_size_x = ax_size
                ax_size_y = ax_size/aspect
            else:
                ax_size_x = ax_size*aspect
                ax_size_y = ax_size

            fig, ax = plot_helpers.fig_fixed_axes((nrows, ncols), (ax_size_y, ax_size_x),
                xlabel='Probe Freq', ylabel='Pump Freq', left_margin=0.7, bot_margin=0.6,
                hspace=0.15, vspace=0.15, padding=0.3)

            if nrows > ncols:
                ax = ax[:, 0]
            else:
                ax = ax[0, :]


        if scale == 'fullmax':
            m = abs(ds.spec2d).max()
        elif scale == 'firstmax':
            m = abs(ds.spec2d[idx[0], ...]).max()
        else:
            raise ValueError("scale must be either fullmax or firstmax")

        if average is not None:
            s2d = uniform_filter1d(ds.spec2d, average, 0, mode="nearest")
        else:
            s2d = ds.spec2d

        if isinstance(ax, plt.Axes):
            ax = [ax]

        for i, k in enumerate(idx):
            c = ax[i].contourf(ds.probe_wn,
                               ds.pump_wn,
                               s2d[k].T,
                               levels=np.linspace(-m, m, 21),
                               cmap='bwr',
                           )
            start = max(ds.probe_wn.min(), ds.pump_wn.min())
            ax[i].axline((start, start), slope=1, c='k', lw=0.5)
            if ds.t[k] < 1:
                title = '%.0f fs' % (ds.t[k] * 1000)
            else:
                title = '%.1f ps' % (ds.t[k])
            ax[i].text(x=0.05, y=0.95, s=title, fontweight='bold', va='top', ha='left',
                       transform=ax[i].transAxes)
        return fig, ax

    def movie_contour(self, fname, contour_kw={}, subplots_kw={}):
        from matplotlib.animation import FuncAnimation

        c, ax = self.contour(self.ds.t[0], **subplots_kw)
        fig = ax.get_figure()
        frames = self.ds.t
        std_kws = {}

        def func(x):
            ax.cla()
            self.contour(x, ax=ax, scale="fullmax", **subplots_kw)

        ani = FuncAnimation(fig=fig, func=func, frames=frames)
        ani.save(fname)

    def plot_cls(self):
        pass

    def plot_square(self, probe_range, pump_range=None, use_symlog=True, ax=None):
        if pump_range is None:
            pump_range = probe_range
        pr = inbetween(self.ds.probe_wn, min(probe_range), max(probe_range))
        reg = self.ds.spec2d[:, pr, :]
        pu = inbetween(self.ds.pump_wn, min(pump_range), max(pump_range))
        reg = reg[:, :, pu]
        s = reg.sum(1).sum(1)
        if ax is None:
            ax = plt.gca()
        l, = ax.plot(self.ds.t, s)
        plot_helpers.lbl_trans(ax, use_symlog)
        return l

    def pump_slice_amps(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.ds.pump_wn, self.ds.pump_slice_amp())
        plot_helpers.ir_mode()
        ax.set(xlabel=plot_helpers.freq_label, ylabel='Slice Amp. [mOD]')

    def elp(self, t, offset=None, p=None):
        ds = self.ds
        spec_i = ds.t_idx(t)
        fig, (ax, ax1) = plt.subplots(2, figsize=(3, 6), sharex='col')

        d = ds.spec2d[spec_i].real.copy()[::, ::].T
        interpol = RegularGridInterpolator((ds.pump_wn, ds.probe_wn,), d[::, ::], bounds_error=False)
        m = abs(d).max()
        ax.pcolormesh(ds.probe_wn, ds.pump_wn, d, cmap='seismic', vmin=-m, vmax=m)

        ax.set(ylim=(ds.pump_wn.min(), ds.pump_wn.max()), xlim=(ds.probe_wn.min(), ds.probe_wn.max()))
        ax.set_aspect(1)
        if offset is None:
            offset = ds.pump_wn[np.argmin(np.min(d, 1))] - ds.probe_wn[np.argmin(np.min(d, 0))]
        if p is None:
            p = ds.probe_wn[np.argmin(np.min(d, 0))]
        y_diag = ds.probe_wn + offset
        y_antidiag = -ds.probe_wn + 2 * p + offset
        ax.plot(ds.probe_wn, y_diag, lw=1)
        ax.plot(ds.probe_wn, y_antidiag, lw=1)

        diag = interpol(np.column_stack((y_diag, ds.probe_wn)))
        antidiag = interpol(np.column_stack((y_antidiag, ds.probe_wn)))
        ax1.plot(ds.probe_wn, diag)
        ax1.plot(ds.probe_wn, antidiag)
        return


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
    plot: 'TwoDimPlotter' = attr.Factory(TwoDimPlotter, True)  # typing: Ignore
    "Plot object offering plotting methods"
    interpolator_: Optional[RegularGridInterpolator] = None  # typing: Ignore

    def _make_int(self):
        intp = RegularGridInterpolator((self.t, self.probe_wn, self.pump_wn),
                                       self.spec2d,
                                       bounds_error=False)
        return intp

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
        self.spec2d = self.spec2d[:, :, i1][:, i2, :]

    def copy(self) -> 'TwoDim':
        """
        Makes a copy of the dataset.
        """
        cpy = attr.evolve(self)
        cpy.plot = TwoDimPlotter(cpy)  # typing: ignore
        return cpy

    def t_idx(self, t: Union[float, Iterable[float]]) -> int:
        """Return nearest idx to nearest time value"""
        return dv.fi(self.t, t)

    def probe_idx(self, wn: Union[float, Iterable[float]]) -> int:
        """Return nearest idx to nearest probe_wn value"""
        return dv.fi(self.probe_wn, wn)

    def pump_idx(self, wn: Union[float, Iterable[float]]) -> int:
        """Return nearest idx to nearest pump_wn value"""
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

    def select_t_range(self, t_min: float = -np.inf, t_max: float = np.inf) -> 'TwoDim':
        """"
        Returns a dataset only containing the data within given time limits.
        """
        idx = inbetween(self.t, t_min, t_max)
        ds = self.copy()
        ds.t = ds.t[idx]
        ds.spec2d = ds.spec2d[idx, :, :]
        return ds

    def integrate_pump(self, lower: float = -np.inf, upper: float = np.inf) -> TimeResSpec:
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

    def single_cls(self,
                   t: float,
                   pr_range: float = 9.0,
                   pu_range: float = 7.0,
                   mode: Literal['neg', 'pos'] = 'neg',
                   method: Literal['com', 'quad', 'fit', 'log_quad'] = 'com'
                   ) -> Tuple[np.ndarray, np.ndarray, object]:
        """
        Calculate the CLS for single 2D spectrum.

        Parameters
        ----------
        t : float
            Delay time of the spectrum to analyse
        pr_range : float, optional
            How many wavenumbers away from the maximum to use for
            determining the exact position, by default 9
        pu_range : float, optional
            The range around the pump-maxima used for calculating
            the CLS.
        mode : ('neg', 'pos'), optional
            negative or positive maximum, by default 'neg'
        method: ('com', 'quad', 'fit')
            Selects the method used for determination of the
            maximum signal. `com` uses the center-of-mass,
            `quad` uses a quadratic fit and `fit` uses
            a gaussian fit.

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
            i = (pr < pr_max + pr_range) & (pr > pr_max - pr_range)
            cen_of_m = np.average(pr[i], weights=s[i])
            if method == 'fit':
                mod = lmfit.models.GaussianModel()
                amp = np.trapz(s[i], pr[i])
                result = mod.fit(s[i], sigma=10, center=cen_of_m, amplitude=amp,
                                 x=pr[i])
                l.append(result.params['center'])
            elif method == 'quad':
                p: Polynomial = Polynomial.fit(pr[i], s[i], 2)  # type: Ignore
                l.append(p.deriv().roots()[0])
            elif method == 'log_quad':
                s_min = s[m]
                i2 = (s < s_min * 0.1)
                p: Polynomial = Polynomial.fit(pr[i & i2], np.log(-s[i & i2]), 2)  # type: Ignore
                l.append(p.deriv().roots()[0])
            else:
                l.append(cen_of_m)

        x = pu[pu_idx]
        y = np.array(l)
        r = linregress(x, y)
        return x, y, r

    def cls(self, **cls_args) -> CLSResult:
        """Calculates the CLS for all 2d-spectra. The arguments are given
        to the single cls function. Returns as `CLSResult`."""
        slopes, slope_errs = [], []
        lines = []
        intercept = []
        intercept_errs = []
        for d in self.t:
            x, y, r = self.single_cls(d, **cls_args)
            slopes.append(r.slope)
            slope_errs.append(r.stderr)
            lines.append(np.column_stack((x, y)))
            intercept.append(r.intercept)
            intercept_errs.append(r.intercept_stderr)
        res = CLSResult(self.t, slopes=slopes, slope_errors=slope_errs,
                        lines=lines, intercepts=intercept, intercept_errors=intercept_errs)
        self.cls_result_ = res
        return res

    def diag_and_antidiag(self, t: float, offset: Optional[float] = None, p: Optional[float] = None) -> DiagResult:
        """
        Extracts the diagonal and anti-diagonal.

        Parameters
        ----------
        t: float
            Waiting time of the 2d-spectra from which the data is extracted.
        offset: float
            Offset of the diagonal, if none, it will we determined by the going through the signal
            minimum.
        p: float
            The point where the anti-diagonal crosses the diagonal. If none, it also goes through
            the signal minimum.

        Returns
        -------
        CLSResult
            Contains the diagonals, coordinates and points.
        """

        spec_i = self.t_idx(t)

        if self.interpolator_ is None:
            self.interpolator_ = self._make_int()

        d = self.spec2d[spec_i, ...].T

        if offset is None:
            offset: float = self.pump_wn[np.argmin(np.min(d, 1))] - self.probe_wn[np.argmin(np.min(d, 0))]
        if p is None:
            p: float = self.probe_wn[np.argmin(np.min(d, 0))]

        y_diag = self.probe_wn + offset
        y_antidiag = -self.probe_wn + 2 * p + offset

        ts = np.ones_like(y_diag)
        diag = self.interpolator_(np.column_stack((ts, self.probe_wn, y_diag)))
        antidiag = self.interpolator_(np.column_stack((ts, self.probe_wn, y_antidiag)))

        res = DiagResult(
            diag=diag,
            antidiag=antidiag,
            diag_coords=y_diag,
            antidiag_coords=y_antidiag,
            offset=offset,
            p=p,
        )
        return res

    def pump_slice_amp(self, t, bg_correct=True):
        d = self.spec2d[self.t_idx(t), :, :]
        diag = np.ptp(d, axis=0)
        if bg_correct:
            diag -= (diag[0] + diag[-1])/2
