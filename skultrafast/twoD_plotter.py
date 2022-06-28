from skultrafast import plot_helpers
from skultrafast.utils import inbetween

import attr
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import RegularGridInterpolator
from typing import Any, TypedDict, Union, Literal, Optional, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from skultrafast.twoD_dataset import TwoDim


class ContourOptions(TypedDict):
    levels: int
    cmap: str
    linewidth: float
    add_lines: bool
    add_diag: bool


@attr.s(auto_attribs=True)
class TwoDimPlotter:
    ds: 'TwoDim' = attr.ib()

    def contour(self, *times,  ax=None, ax_size: float = 1.5, subplots_kws={}, aspect=None, labels={'x': "Probe Freq. [cm-1]", 'y': "Pump Freq. [cm-1]"},
                direction: Union[Tuple[int, int], str] = 'vertical', contour_params: dict = {},
                scale: Literal['firstmax', 'fullmax', 'eachmax'] = "firstmax", average=None, fig_kws: dict = {}) -> dict[str, Any]:
        ds = self.ds
        idx = [ds.t_idx(i) for i in times]
        if ax is None:
            if aspect is None:
                aspect = ds.probe_wn.ptp() / ds.pump_wn.ptp()
            if direction[0] == 'v':
                nrows = len(idx)
                ncols = 1
            elif direction[0] == 'h':
                nrows = 1
                ncols = len(idx)
            else:
                nrows = direction[0]
                ncols = direction[1]

            if aspect > 1:
                ax_size_x = ax_size
                ax_size_y = ax_size/aspect
            else:
                ax_size_x = ax_size*aspect
                ax_size_y = ax_size

            fig, ax = plot_helpers.fig_fixed_axes((nrows, ncols), (ax_size_y, ax_size_x),
                                                  xlabel=labels['x'], ylabel=labels['y'], left_margin=0.7, bot_margin=0.6,
                                                  hspace=0.15, vspace=0.15, padding=0.3, **fig_kws)

            if nrows > ncols:
                ax = ax[:, 0]
            else:
                ax = ax[0, :]

        if scale == 'fullmax':
            m = abs(ds.spec2d).max()
        elif scale == 'firstmax':
            m = abs(ds.spec2d[idx[0], ...]).max()
        elif scale == 'eachmax':
            m = None
        else:
            raise ValueError("scale must be either fullmax or firstmax")

        if average is not None:
            s2d = uniform_filter1d(ds.spec2d, average, 0, mode="nearest")
        else:
            s2d = ds.spec2d

        if isinstance(ax, plt.Axes):
            ax = [ax]

        contour_ops: ContourOptions = ContourOptions(
            levels=21,
            cmap='bwr',
            linewidth=0.5,
            add_lines=True,
            add_diag=True
        )
        contour_ops.update(contour_params)
        out = {'fig': fig, 'axs': ax}
        for i, k in enumerate(idx):
            out_i = {'ax': ax[i]}
            if scale == 'eachmax':
                m = np.abs(s2d[k, ...]).max()

            levels = np.linspace(-m, m, contour_ops['levels'])
            c = ax[i].contourf(ds.probe_wn,
                               ds.pump_wn,
                               s2d[k].T,
                               levels=levels,
                               cmap=contour_ops['cmap'],
                               )
            out_i['contourf'] = c

            if contour_ops['add_lines']:
                cl = ax[i].contour(ds.probe_wn,
                                   ds.pump_wn,
                                   s2d[k].T,
                                   levels=levels,
                                   colors='k',
                                   linestyles='-',
                                   linewidths=contour_ops['linewidth'],
                                   )
                out_i['contour'] = cl
            if contour_ops['add_diag']:
                start = max(ds.probe_wn.min(), ds.pump_wn.min())
                out_i['diag_line'] = ax[i].axline((start, start), slope=1, c='k', lw=0.5)
            if ds.t[k] < 1:
                title = '%.0f fs' % (ds.t[k] * 1000)
            else:
                title = '%.1f ps' % (ds.t[k])
            out_i['title'] = ax[i].text(x=0.05, y=0.95, s=title, fontweight='bold', va='top', ha='left',
                                        transform=ax[i].transAxes)
            out[i] = out_i
        return out

    def single_contour(self, t, co: ContourOptions = ContourOptions(), ax=None) -> dict:
        if ax is None:
            ax = plt.gca()

        ds = self.ds
        s2d = ds.spec2d[ds.t_idx(t)]
        m = abs(s2d).max()
        levels = np.linspace(-m, m, co.levels)
        out = {"ax": ax}
        c = ax.contourf(ds.probe_wn,
                        ds.pump_wn,
                        s2d.T,
                        levels=levels,
                        cmap=co.cmap,
                        )
        out = {"contourf": c}
        if co.add_line:
            cl = ax.contour(ds.probe_wn,
                            ds.pump_wn,
                            s2d.T,
                            levels=levels,
                            colors='k',
                            linestyles='-',
                            linewidths=co.linewidth,
                            )
            out['contour'] = cl
        if co.add_diag:
            start = max(ds.probe_wn.min(), ds.pump_wn.min())
            out['diag'] = ax.axline((start, start), slope=1, c='k', lw=0.5)
        return out

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

    def plot_square(self, pump_range, probe_range=None, use_symlog=True, ax=None):
        if probe_range is None:
            probe_range = pump_range
        pr = inbetween(self.ds.probe_wn, min(probe_range), max(probe_range))
        reg = self.ds.spec2d[:, pr, :]
        pu = inbetween(self.ds.pump_wn, min(pump_range), max(pump_range))
        reg = reg[:, :, pu]
        s = reg.sum(1).sum(1)
        if ax is None:
            ax = plt.gca()
        l, = ax.plot(self.ds.t, s)
        if symlog:
            ax.set_xscale("symlog", linthresh=1.0, linscale=linscale)
        plot_helpers.lbl_trans(ax, use_symlog)
        return l

    def elp(self, t, offset=None, p=None):
        ds = self.ds
        spec_i = ds.t_idx(t)
        fig, (ax, ax1) = plt.subplots(2, figsize=(3, 6), sharex='col')

        d = ds.spec2d[spec_i].real.copy()[::, ::].T
        interpol = RegularGridInterpolator(
            (ds.pump_wn, ds.probe_wn,), d[::, ::], bounds_error=False)
        m = abs(d).max()
        ax.pcolormesh(ds.probe_wn, ds.pump_wn, d, cmap='seismic', vmin=-m, vmax=m)

        ax.set(ylim=(ds.pump_wn.min(), ds.pump_wn.max()),
               xlim=(ds.probe_wn.min(), ds.probe_wn.max()))
        ax.set_aspect(1)
        if offset is None:
            offset = ds.pump_wn[np.argmin(np.min(d, 1))] - \
                ds.probe_wn[np.argmin(np.min(d, 0))]
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

    def psa(self, t: float, bg_correct: bool = True,
            normalize: Optional[Union[float, Literal['max']]] = None, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ds = self.ds
        diag = ds.pump_slice_amp(t, bg_correct=bg_correct)

        if normalize is None:
            pass
        elif normalize == 'max':
            diag /= diag.max()
        else:
            diag = diag/diag[ds.pump_idx(normalize)]

        kwargs.update(label='%d ps' % t)
        line = ax.plot(ds.pump_wn, diag, **kwargs)
        ax.set(xlabel='Pump Freq.', ylabel='Slice Amplitude')
        return line
