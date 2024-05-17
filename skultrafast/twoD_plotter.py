from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, TypedDict,
                    Union)

import attr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter1d

from skultrafast import plot_helpers
from skultrafast.utils import inbetween

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

    def contour(self,
                *times,
                ax=None,
                ax_size: float = 1.5,
                subplots_kws={},
                aspect=None,
                labels={
                    'x': "Probe Freq. [cm-1]",
                    'y': "Pump Freq. [cm-1]"
                },
                direction: Union[Tuple[int, int], Literal['v', 'h']] = 'v',
                contour_params: Dict[str, Any] = {},
                scale: Literal['firstmax', 'fullmax', 'eachmax'] = "firstmax",
                average=None,
                fig_kws: dict = {}) -> Dict[Union[str, int], Any]:
        ds = self.ds
        idx = [ds.t_idx(i) for i in times]
        if ax is None:
            if aspect is None:
                aspect = ds.probe_wn.ptp() / ds.pump_wn.ptp()
            if direction[0] == 'v':
                nrows = len(idx)
                ncols = 1
            elif isinstance(direction, tuple):
                nrows: int = direction[0]
                ncols: int = direction[1]
            else:
                nrows = 1
                ncols = len(idx)

            if aspect > 1:
                ax_size_x = ax_size
                ax_size_y = ax_size / aspect
            else:
                ax_size_x = ax_size * aspect
                ax_size_y = ax_size

            fig, ax = plot_helpers.fig_fixed_axes(
                (nrows, ncols),
                (ax_size_y, ax_size_x),  # typing: ignore
                xlabel=labels['x'],
                ylabel=labels['y'],
                left_margin=0.7,
                bot_margin=0.6,
                hspace=0.15,
                vspace=0.15,
                padding=0.3,
                **fig_kws)

            if nrows > ncols:
                ax = ax[:, 0]
            else:
                ax = ax[0, :]
        else:
            fig = ax[0].get_figure()

        if average is not None:
            s2d = uniform_filter1d(ds.spec2d, average, 0, mode="nearest")
        else:
            s2d = ds.spec2d

        if scale == 'fullmax':
            m = abs(s2d).max()
        elif scale == 'firstmax':
            m = abs(s2d[idx[0], ...]).max()
        elif scale == 'eachmax':
            m = None
        else:
            raise ValueError("scale must be either fullmax or firstmax")

        if isinstance(ax, plt.Axes):
            ax = [ax]

        contour_ops: ContourOptions = ContourOptions(levels=20,
                                                     cmap='bwr',
                                                     linewidth=0.5,
                                                     add_lines=True,
                                                     add_diag=True)
        contour_ops.update(contour_params)
        out = {'fig': fig, 'axs': ax}
        for i, k in enumerate(idx):
            out_i = {'ax': ax[i]}

            if scale == 'eachmax':
                m = np.abs(s2d[k, ...]).max()
            if isinstance(contour_ops['levels'], int):
                assert m is not None
                levels = np.linspace(-m, m, contour_ops['levels'])
            else:
                levels = np.array(contour_ops['levels'])
            c = ax[i].contourf(
                ds.probe_wn,
                ds.pump_wn,
                s2d[k].T,
                levels=levels,
                cmap=contour_ops['cmap'],
            )
            out_i['contourf'] = c

            if contour_ops['add_lines']:
                cl = ax[i].contour(
                    ds.probe_wn,
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
            out_i['title'] = ax[i].text(x=0.05,
                                        y=0.95,
                                        s=title,
                                        fontweight='bold',
                                        va='top',
                                        ha='left',
                                        transform=ax[i].transAxes)
            out[i] = out_i
        return out

    def single_contour(self, t, co: ContourOptions = ContourOptions(), ax=None) -> dict:
        if ax is None:
            ax = plt.gca()
        contour_ops: ContourOptions = ContourOptions(levels=20,
                                                     cmap='bwr',
                                                     linewidth=0.5,
                                                     add_lines=True,
                                                     add_diag=True)
        contour_ops.update(co)
        ds = self.ds
        s2d = ds.spec2d[ds.t_idx(t)]
        m = abs(s2d).max()
        levels = np.linspace(-m, m, contour_ops.levels)
        out = {"ax": ax}
        c = ax.contourf(
            ds.probe_wn,
            ds.pump_wn,
            s2d.T,
            levels=levels,
            cmap=contour_ops.cmap,
        )
        out = {"contourf": c}
        if co.add_line:
            cl = ax.contour(
                ds.probe_wn,
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

        out = self.single_contour(self.ds.t[0], **subplots_kw)
        ax = out['ax']
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

    def plot_square(self,
                    pump_range: Tuple[float, float],
                    probe_range: Optional[Tuple[float, float]] = None,
                    symlog: bool = True,
                    ax: Optional[plt.Axes] = None,
                    mode: Literal['trapz', 'sum', 'ptp', 'min', 'max'] = 'trapz',
                    **plot_kws):
        """
        Plot the integrated signal of given region over the waiting time.

        Parameters
        ----------
        pump_range: Tuple[float, float]
            The range of the pump frequency in cm-1
        probe_range: Tuple[float, float]
            The range of the probe frequency in cm-1. If not given uses the same
            as the pump range.
        symlog: bool
            If True, use symlog scale for the time axis.
        ax: plt.Axes
            The axes to plot on. If None, the current axes is used.
        mode: str
            The mode of signal calculation. Can be either 'trapz' or 'ptp'.


        Returns
        -------
        l: plt.Line2D
            The line objects created by the plot.
        """

        if ax is None:
            ax = plt.gca()

        if probe_range is None:
            probe_range = pump_range
        pr = inbetween(self.ds.probe_wn, min(probe_range), max(probe_range))
        reg = self.ds.spec2d[:, pr, :]
        pu = inbetween(self.ds.pump_wn, min(pump_range), max(pump_range))
        reg = reg[:, :, pu]
        if mode == 'trapz':
            s = np.trapz(reg, self.ds.pump_wn[pu], axis=2)
            s = np.trapz(s, self.ds.probe_wn[pr], axis=1)
        elif mode == 'sum':
            s = np.sum(reg, axis=2)
            s = np.sum(s, axis=1)
        elif mode == 'ptp':
            s = np.ptp(reg, axis=1)
            s = np.max(s, axis=1)
        elif mode == 'min':
            s = np.min(reg, axis=2)
            s = np.min(s, axis=1)
        elif mode == 'max':
            s = np.max(reg, axis=2)
            s = np.max(s, axis=1)
        assert ax is not None
        l, = ax.plot(self.ds.t, s, **plot_kws)
        if symlog:
            ax.set_xscale("symlog", linthresh=1.0, linscale=1)
        plot_helpers.lbl_trans(ax, symlog)
        return l

    def elp(self, t, offset=None, p=None):
        ds = self.ds
        spec_i = ds.t_idx(t)
        fig, (ax, ax1) = plt.subplots(2, figsize=(3, 6), sharex='col')

        d = ds.spec2d[spec_i].real.copy()[::, ::].T
        interpol = RegularGridInterpolator((
            ds.pump_wn,
            ds.probe_wn,
        ),
            d[::, ::],
            bounds_error=False)
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
        y_antidiag = -ds.probe_wn + 2*p + offset
        ax.plot(ds.probe_wn, y_diag, lw=1)
        ax.plot(ds.probe_wn, y_antidiag, lw=1)

        diag = interpol(np.column_stack((y_diag, ds.probe_wn)))
        antidiag = interpol(np.column_stack((y_antidiag, ds.probe_wn)))
        ax1.plot(ds.probe_wn, diag)
        ax1.plot(ds.probe_wn, antidiag)
        return

    def psa(self,
            t: float,
            bg_correct: bool = True,
            normalize: Optional[Union[float, Literal['max']]] = None,
            ax: Optional[plt.Axes] = None,
            **kwargs):
        """
        Plot the pump-slice amplitude spectrum for a given waiting time.

        Parameters
        ----------
        t : float
            Waiting time in ps.
        bg_correct : bool, optional
            Whether to subtract a constant background, by default True.
        normalize : Optional[Union[float, Literal['max']]], optional
            Whether to normalize the spectrum. If a float is given, the spectrum is divided by
            the value at that pump frequency. If 'max' is given, the spectrum is divided by its
            maximum value, by default not normalized.
        ax : Optional[matplotlib.axes.Axes], optional
            The axes to plot on. If None, the current axes is used.
        """
        if ax is None:
            ax = plt.gca()
        ds = self.ds
        diag = ds.pump_slice_amp(t, bg_correct=bg_correct)

        if normalize is None:
            pass
        elif normalize == 'max':
            diag /= diag.max()
        else:
            diag = diag / diag[ds.pump_idx(normalize)]

        kwargs.update(label='%d ps' % t)
        line = ax.plot(ds.pump_wn, diag, **kwargs)
        ax.set(xlabel='Pump Freq.', ylabel='Slice Amplitude')
        return line

    def diagonal(self,
                 *t: float,
                 offset: Optional[float] = None,
                 ax: Optional[plt.Axes] = None) -> List[plt.Line2D]:
        """
        Plots the signal along the diagonal. If offset is not None, the signal
        is shifted by offset, otherwise the offset is determined from the signal
        by shifting it to the maximal signal.

        Parameters
        ----------
        t : float
            The waiting time
        offset : float, optional
            The offset from the signal, is the spectra is calibrated should be
            0. If None, the offset is determined from the signal by looking for
            maximal signal.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, the current axes is used.

        Returns
        -------
        List[matplotlib.lines.Line2D]
            The plotted lines objects
        """
        if ax is None:
            ax = plt.gca()
        l = []
        for ti in t:
            diag_data = self.ds.diag_and_antidiag(ti, offset)
            l += ax.plot(diag_data.diag_coords, diag_data.diag, label='%.1f ps' % ti)
        ax.set(xlabel=plot_helpers.freq_label, ylabel='Diagonal Amplitude [AU]')
        return l

    def anti_diagonal(self,
                      *t: float,
                      offset: Optional[float] = None,
                      p: Optional[float] = None,
                      ax: Optional[plt.Axes] = None) -> List[plt.Line2D]:
        """
        Plots the signal along a antidiagonal. If offset is not None, the digonal
        goes through the to the maximal signal. If the frequency axes are
        calibrated, the offset should be 0. p deterimines the position where the
        anti-diagonal goes through. If None, again the position of the maxium is
        used.

        Parameters
        ----------
        t : float
            The waiting time
        offset : float, optional
            The offset from the signal, is the spectra is calibrated should be
            0. If None, the offset is determined from the signal by looking for
            maximal signal.
        p: float, optional
            The position of the anti-diagonal. If None, the position of the
            maximal signal is used.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, the current axes is used.

        Returns
        -------
        List[matplotlib.lines.Line2D]
            The plotted lines objects
        """
        if ax is None:
            ax = plt.gca()
        l = []
        for ti in t:
            diag_data = self.ds.diag_and_antidiag(ti, offset, p)
            l += ax.plot(diag_data.antidiag_coords,
                         diag_data.antidiag,
                         label='%.1f ps' % ti)
        ax.set(xlabel=plot_helpers.freq_label, ylabel='Anti-diagonal Amplitude [AU]')
        return l

    def mark_minmax(self,
                    t: float,
                    which: Literal['both', 'min', 'max'] = 'both',
                    ax: Optional[plt.Axes] = None,
                    **kwargs):
        """
        Marks the position of the minimum and maxium of a given time t.
        """

        if ax is None:
            ax = plt.gca()
        minmax = self.ds.get_minmax(t)
        points = []

        if which in ['both', 'min']:
            points += [(minmax['ProbeMin'], minmax['PumpMin'])]
        if which in ['both', 'max']:
            points += [(minmax['ProbeMax'], minmax['PumpMax'])]
        plotkws = {
            'color': 'yellow',
            'marker': '+',
            'ls': 'none',
            'markersize': 9,
            **kwargs
        }
        return ax.plot(*zip(*points), **plotkws)

    def trans(self,
              pump_wn: Union[float, list[float]],
              probe_wn: Union[float, list[float]],
              ax: Optional[plt.Axes] = None,
              symlog=True,
              **kwargs) -> List[plt.Line2D]:
        """
        Plot the 2D signal of single point over the waiting time.

        Parameters
        ----------
        pump_wn : float or list of float
            The pump frequency. Also takes a list. If a list is given, the length
            of the list must be the same as the length of probe_wn or of length 1.
        probe_wn : float or list of float
            The probe frequency. Also takes a list. If a list is given, the length
            of the list must be the same as the length of pump_wn or of length 1.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, the current axes is used.
        symlog : bool, optional
            If True, apply symlog scaling to the plot.
        kwargs : dict
            Additional keyword arguments are passed to the plot function.
        Returns
        -------
        List[matplotlib.lines.Line2D]
            The plotted lines objects
        """
        if ax is None:
            ax = plt.gca()
        if isinstance(pump_wn, (float, int)):
            pump_wn = [pump_wn]
        if isinstance(probe_wn, (float, int)):
            probe_wn = [probe_wn]

        if not len(pump_wn) == len(probe_wn):
            if len(pump_wn) == 1:
                pump_wn = pump_wn * len(probe_wn)
            elif len(probe_wn) == 1:
                probe_wn = probe_wn * len(pump_wn)
            else:
                raise ValueError(
                    'The length of pump_wn and probe_wn must be the same or one of them must be 1.'
                )
        l = []
        for x, y in zip(pump_wn, probe_wn):
            dat = self.ds.data_at(pump_wn=x, probe_wn=y)
            l += ax.plot(self.ds.t, dat, label='%.1f, %.1f' % (x, y), **kwargs)
        if symlog:
            ax.set_xscale('symlog', linthresh=1)
        plot_helpers.lbl_trans(ax, use_symlog=symlog)
        return l
