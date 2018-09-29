import numpy as np
import astropy.stats as stats
from skultrafast.data_io import save_txt
from skultrafast.filter import bin_channels, uniform_filter

from skultrafast import zero_finding
import skultrafast.plot_helpers as ph
import skultrafast.dv as dv
from enum import Enum
from types import SimpleNamespace
from collections import namedtuple
import matplotlib.pyplot as plt


class MesspyDataSet:
    def __init__(self, fname, is_pol_resolved=False,
                 pol_first_scan='unknown', valid_channel='both'):
        """Class for working with data files from MessPy.

        Parameters
        ----------
        fname : str
            Filename to open.
        is_pol_resolved : bool (false)
            If the dataset was recorded polarization resolved.
        pol_first_scan : {'magic', 'para', 'perp', 'unknown'}
            Polarization between the pump and the probe in the first scan. If
            `valid_channel` is 'both', this corresponds to the zeroth channel.
        valid_channel : `0`, `1`, 'both'
            Indicates which channels contains a real signal. For recently
            recorded data, it is 0 for the visible setup and 1 for the IR
            setup. Older IR data uses both.
        """

        with np.load(fname) as f:
            self.wl = f['wl']
            self.t = f['t']
            self.data = f['data']

        self.pol_first_scan = pol_first_scan
        self.is_pol_resolved = is_pol_resolved
        self.valid_channel = valid_channel

    def average_scans(self, sigma=3):
        """
        Calculate the average of the scans. Uses sigma clipping, which
        also filters nans. For polarization resolved measurements, the
        function assumes that the polarisation switches every scan.

        Parameters
        ----------
        sigma : float
           sigma used for sigma clipping.

        Returns
        -------
        dict or DataSet
            DataSet or Dict of DataSets containing the averaged datasets.

        """

        num_wls = self.data.shape[0]

        if not self.is_pol_resolved:
            data = stats.sigma_clip(self.data,
                                    sigma=sigma, axis=-1)
            mean = data.mean(-1)
            std = data.std(-1)
            err = std / np.sqrt(data.mask.sum(-1))

            if self.valid_channel in [0, 1]:
                mean = data[..., self.valid_channel]
                std = std[..., self.valid_channel]
                err = err[..., self.valid_channel]

                out = {}

                if num_wls > 1:
                    for i in range(num_wls):
                        ds = DataSet(self.wl[:, i], self.t, mean[i, ...],
                                     err[i, ...])
                        out[self.pol_first_scan + str(i)] = ds
                else:
                    out = DataSet(self.wl[:, 0], self.t, mean[0, ...],
                                  err[0, ...])
                return out

        elif self.is_pol_resolved and self.valid_channel in [0, 1]:
            assert (self.pol_first_scan in ['para', 'perp'])
            data1 = stats.sigma_clip(self.data[..., self.valid_channel, ::2],
                                     sigma=sigma, axis=-1)
            mean1 = data1.mean(-1)
            std1 = data1.std(-1)
            err1 = std1 / np.sqrt(data1.mask.sum(-1))

            data2 = stats.sigma_clip(self.data[..., self.valid_channel, 1::2],
                                     sigma=sigma, axis=-1)
            mean2 = data2.mean(-1)
            std2 = data2.std(-1)
            err2 = std2 / np.sqrt(data2.mask.sum(-1))

            out = {}
            for i in range(self.data.shape[0]):
                out[self.pol_first_scan + str(i)] = DataSet(self.wl[:, i],
                                                            self.t,
                                                            mean1[i, ...],
                                                            err1[i, ...])
                other_pol = 'para' if self.pol_first_scan == 'perp' else 'perp'
                out[other_pol + str(i)] = DataSet(self.wl[:, i], self.t,
                                                  mean2[i, ...], err2[i, ...])
                iso = 1 / 3 * out['para' + str(i)].data + 2 / 3 * out[
                    'perp' + str(i)].data
                iso_err = np.sqrt(
                    1 / 3 * out['para' + str(i)].err ** 2 + 2 / 3 * out[
                        'perp' + str(i)].data ** 2)
                out['iso' + str(i)] = DataSet(self.wl[:, i], self.t, iso,
                                              iso_err)
            return out
        else:
            raise NotImplementedError("Iso correction not suppeorted yet.")


EstDispResult = namedtuple('EstDispResult', 'correct_ds tn polynomial')
EstDispResult.__doc__ = """
Tuple containing the results from an dispersion estimation.

Attributes
----------
correct_ds : DataSet
    A dataset were we used linear interpolation to remove the dispersion.
tn : array
    Array containing the results of the applied heuristic. 
polynomial : function
    Function which maps wavenumbers to time-zeros.
"""



class DataSet:
    def __init__(self, wl, t, data, err=None, name=None, freq_unit='nm'):
        """
        Class for working with time-resolved spectra. If offers methods for
        analyzing and pre-processing the data. To visualize the data,
        each `DataSet` object has an instance of an `DataSetPlotter` object
        accessible under `plot`.

        Parameters
        ----------
        wl : array of shape(n)
            Array of the spectral dimension
        t : array of shape(m)
            Array with the delay times.
        data : array of shape(n, m)
            Array with the data for each point.
        err : array of shape(n, m) or None (optional)
            Contains the std err of the data, can be `None`.
        name : str
            Identifier for data set. (optional)
        freq_unit : 'nm' or 'cm'
            Unit of the wavelength array, default is 'nm'.

        Attributes
        ----------
        plot : DataSetPlotter
            Helper class which can plot the dataset using `matplotlib`.

        """

        assert ((t.shape[0], wl.shape[0]) == data.shape)

        if freq_unit == 'nm':
            self.wavelengths = wl
            self.wavenumbers = 1e7 / wl
            self.wl = self.wavelengths
        else:
            self.wavelengths = 1e7 / wl
            self.wavenumbers = wl
            self.wl = self.wavenumbers

        self.t = t
        self.data = data
        self.err = err

        if name is not None:
            self.name = name

        # Sort wavelenths and data.
        idx = np.argsort(self.wavelengths)
        self.wavelengths = self.wavelengths[idx]
        self.wavenumbers = self.wavenumbers[idx]
        self.data = self.data[:, idx]
        self.plot = DataSetPlotter(self)

    def __iter__(self):
        """For compatbility with dv.tup"""
        return iter(self.wavelengths, self.t, self.data)

    @classmethod
    def from_txt(cls, fname, freq_unit='nm', time_div=1., loadtxt_kws=None):
        """
        Directly create a dataset from a text file.

        Parameters
        ----------
        fname : str
            Name of the file. This function assumes the data is given by a
            (n+1, m+1) table. Excludig the [0, 0] value, the first row gives the
            frequencies and the first column gives the delay-times.
        freq_unit : {'nm', 'cm'}
            Unit of the frequencies.
        time_div : float
            Since `skultrafast` prefers to work with picoseconds and programs
            may use different units, it divides the time-values by `time_div`.
            Use `1`, the default, to not change the time values.
        loadtxt_kws : dict
            Dict containing keyword arguments to `np.loadtxt`.
        """
        if loadtxt_kws is None:
            loadtxt_kws = {}
        tmp = np.loadtxt(fname, **loadtxt_kws)
        t = tmp[1:, 0] / time_div
        freq = tmp[0, 1:]
        data = tmp[1:, 1:]
        return cls(freq, t, data, freq_unit=freq_unit)

    def save_txt(self, fname, freq_unit='wl'):
        """
        Saves the dataset as a text file.

        Parameters
        ----------
        fname : str
            Filename (can include filepath)

        freq_unit : 'nm' or 'cm' (default 'nm')
            Which frequency unit is used.
        """
        wl = self.wavelengths if freq_unit is 'wl' else self.wavenumbers
        save_txt(fname, wl, self.t, self.data)

    def cut_freqs(self, freq_ranges=None, invert_sel=False, freq_unit='nm'):
        """
        Removes channels inside (or outside ) of given frequency ranges.

        Parameters
        ----------
        freq_ranges : list of (float, float)
            List containing the edges (lower, upper) of the
            frequencies to keep.
        invert_sel : bool
            Invert the final selection.
        freq_unit : {'nm', 'cm'}
            Unit of the given edges.

        Returns
        -------
        : DataSet
            DataSet containing only the listed regions.
        """
        idx = np.zeros_like(self.wavelengths, dtype=np.bool)
        arr = self.wavelengths if freq_unit is 'nm' else self.wavenumbers
        for (lower, upper) in freq_ranges:
            idx ^= np.logical_and(arr > lower, arr < upper)
        if not invert_sel:
            idx = ~idx
        if self.err is not None:
            err = self.err[:, idx]
        else:
            err = None
        return DataSet(arr[idx], self.t, self.data[:, idx], err, freq_unit)

    def mask_freqs(self, freq_ranges=None, invert_sel=False, freq_unit='nm'):
        """
        Mask channels inside of given frequency ranges.

        Parameters
        ----------
        freq_ranges : list of (float, float)
            List containing the edges (lower, upper) of the
            frequencies to keep.
        invert_sel : bool
            When True, it inverts the selection. Can be used
            mark everything outside selected ranges.
        freq_unit : {'nm', 'cm'}
            Unit of the given edges.

        Returns
        -------
        : DataSet
            DataSet containing only the listed regions.
        """
        idx = np.zeros_like(self.wavelengths, dtype=np.bool)
        arr = self.wavelengths if freq_unit is 'nm' else self.wavenumbers

        for (lower, upper) in freq_ranges:
            idx ^= np.logical_and(arr > lower, arr < upper)
        if not invert_sel:
            idx = ~idx
        if self.err is not None:
            self.err.mask[:, idx] = True
        self.data = np.ma.MaskedArray(self.data)
        self.data[:, idx] = np.ma.masked
        # self.wavelengths = np.ma.MaskedArray(self.wavelengths, idx)
        # self.wavenumbers = np.ma.MaskedArray(self.wavenumbers, idx)

    def cut_times(self, time_ranges, invert_sel):
        """
        Remove spectra inside (or outside) of given time-ranges.

        Parameters
        ----------
        time_ranges : list of (float, float)
            List containing the edges of the time-regions to keep.
        invert_sel : bool
            Inverts the final selection.
        Returns
        -------
        : DataSet
            DataSet containing only the requested regions.
        """
        idx = np.zeros_like(self.t, dtype=np.bool)
        arr = self.t
        for (lower, upper) in time_ranges:
            idx ^= np.logical_and(arr > lower, arr < upper)
        if not invert_sel:
            idx = ~idx
        if self.err is not None:
            err = self.err[idx, :]
        else:
            err = None
        return DataSet(self.wavelengths, self.t[idx], self.data[idx, :], err)

    def mask_times(self, time_ranges, invert_sel=False):
        """
        Mask spectra inside (or outside) of given time-ranges.

        Parameters
        ----------
        time_ranges : list of (float, float)
            List containing the edges of the time-regions to keep.
        invert_sel : bool
            Invert the selection.

        Returns
        -------
        : None
        """
        idx = np.zeros_like(self.t, dtype=np.bool)
        arr = self.t
        for (lower, upper) in time_ranges:
            idx ^= np.logical_and(arr > lower, arr < upper)
        if not invert_sel:
            idx = ~idx
        if self.err is not None:
            self.err[idx, :].mask = True
        # self.t = np.ma.MaskedArray(self.t, idx)
        self.data.mask[:, idx] = True

    def subtract_background(self, n: int = 10):
        """Subtracts the first n-spectra from the dataset"""
        self.data -= np.mean(self.data[:n, :], 0, keepdims=1)

    def bin_freqs(self, n: int, freq_unit='nm'):
        """
        Bins down the dataset by averaging over several transients.

        Parameters
        ----------
        n : int
            The number of bins. The edges are calculated by
            np.linspace(freq.min(), freq.max(), n+1).
        freq_unit : {'nm', 'cm'}
            Whether to calculate the bin-borders in
            frequency- of wavelength-space.
        Returns
        -------
        DataSet
            Binned down `DataSet`
        """
        # We use the negative of the wavenumbers to make the array sorted
        arr = self.wavelengths if freq_unit is 'nm' else -self.wavenumbers
        # Slightly offset edges to include themselves.
        edges = np.linspace(arr.min() - 0.002, arr.max() + 0.002, n + 1)
        idx = np.searchsorted(arr, edges)
        binned = np.empty((self.data.shape[0], n))
        binned_wl = np.empty(n)
        for i in range(n):
            binned[:, i] = np.average(self.data[:, idx[i]:idx[i + 1]], 1)
            binned_wl[i] = np.mean(arr[idx[i]:idx[i + 1]])
        if freq_unit is 'cm':
            binned_wl = - binned_wl
        return DataSet(binned_wl, self.t, binned, freq_unit)

    def estimate_dispersion(self, heuristic='abs', heuristic_args=(1,), deg=2,
                            t_parameter=None):
        """
        Estimates the dispersion from a dataset by first
        applying a heuristic to each channel. The results are than
        robustly fitted with a polynomial of given order.

        Parameters
        ----------
        heuristic : {'abs', 'diff', 'gauss_diff'} or func
            Determines which heuristic to use on each channel. Can
            also be a function which follows `func(t, y, *args) and returns
            a `t0`-value. The heuristics are described in `zero_finding`.
        heuristic_args : tuple
            Arguments which are given to the heuristic.
        deg : int (optional)
            Degree of the polynomial used to fit the dispersion (defaults to 2).
        t_parameter : float
            Determines the robustness of the fit. See statsmodels documentation
            for more info.

        Returns
        -------
        EstDispResult
            Tuple containing the dispersion corrected version of the dataset, an
            array with time-zeros from the heuristic, and the polynomial
            function resulting from the robust fit.
        """

        if heuristic == 'abs':
            idx = zero_finding.use_first_abs(self.data, heuristic_args[0])

        vals, coefs = zero_finding.robust_fit_tz(self.wavenumbers, self.t[idx],
                                                 deg, t=t_parameter)
        func = np.poly1d(coefs)
        new_data = zero_finding.interpol(self, func(self.wavenumbers))
        return EstDispResult(
            correct_ds=DataSet(self.wavelengths, self.t, new_data.data),
            tn=self.t[idx], polynomial=func)

    def fit_das(self, x0, fit_t0=False, fix_last_decay=True):
        """
        Fit a sum of expontials to the dataset. This function assumes
        the dataset is already corrected for dispersion.

        Parameters
        ----------
        x0 : list of floats or array
            Starting values of the fit. The first value is the estimate of the
            system response time omega. If `fit_t0` is true, the second float is
            the guess of the time-zero. All other floats are interpreted as the
            guessing values for exponential decays.
        fit_t0 : bool (optional)
            If the time-zero should be determined by the fit too.
        fix_last_decay : bool (optional)
            Fixes the value of the last tau of the initial guess. It can be
            used to add a constant by setting the last tau to a large value
            and fix it.
        """
        pass

    def lft_density_map(self, taus, alpha=1e-4, ):
        """Calculates the LDM from a dataset by regularized regression.

        Parameters
        """
        pass


class DataSetPlotter:
    def __init__(self, dataset: DataSet, freq_unit='nm'):
        """
        Class which can Plot a `DataSet` using matplotlib.

        Parameters
        ----------
        dataset : DataSet
            The DataSet to work with.
        freq_unit : {'nm', 'cm'} (optional)
            The default unit of the plots. To change
            the unit afterwards, set the attribute directly.
        """
        self.dataset = dataset
        self.freq_unit = freq_unit

    def map(self, symlog=True, equal_limits=True,
            plot_con=True, con_step=None, con_filter=None, ax=None,
            **kwargs):
        """
        Plot a colormap of the dataset with optional contour lines.

        Parameters
        ----------
        symlog : bool
            Determines if the yscale is symmetric logarithmic.
        equal_limits : bool
            If true, it makes to colors symmetric around zeros. Note this
            also sets the middle of the colormap to zero.
            Default is `True`.
        plot_con : bool
            Plot additional contour lines if `True` (default).
        con_step : float, array or None
            Controls the contour-levels. If `con_step` is a float, it is used as
            the step size between two levels. If it is an array, its elements
            are the levels. If `None`, it defaults to 20 levels.
        con_filter : None, int or `DataSet`.
            Since contours are strongly affected by noise, it can be prefered to
            filter the dataset before calculating the contours. If `con_filter`
            is a dataset, the data of that set will be used for the contours. If
            it is a int or tuple or int, the data will be filtered with an
            uniform filter before calculation the contours. If `None`, no data
            prepossessing will be applied.
        ax : plt.Axis or None
            Takes a matplotlib axis. If none, it uses `plt.gca()` to get the
            current axes. The lines are plotted in this axis.

        """
        if ax is None:
            ax = plt.gca()
        is_nm = self.freq_unit is 'nm'
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()

        ds = self.dataset
        x = ds.wavelengths if is_nm else ds.wavenumbers
        cmap = kwargs.pop('colormap', "bwr")
        if equal_limits:
            m = np.max(np.abs(ds.data))
            vmin, vmax = -m, m
        else:
            vmin, vmax = ds.max(), ds.min()
        mesh = ax.pcolormesh(x, ds.t, ds.data, vmin=vmin,
                             vmax=vmax, cmap=cmap, **kwargs)
        if symlog:
            ax.set_yscale('symlog', linthreshy=1)
            ph.symticks(ax, axis='y')
            ax.set_ylim(-.5)
        plt.colorbar(mesh, ax=ax)

        if plot_con:
            if con_step is None:
                levels = 20
            elif isinstance(con_step, np.ndarray):
                levels = con_step
            else:
                # TODO This assumes data has positive and negative elements.
                pos = np.arange(0, ds.data.max(), con_step)
                neg = np.arange(0, -ds.data.min(), con_step)
                levels = np.hstack((-neg[::-1][:-1], pos))

            if isinstance(con_filter, DataSet):
                data = con_filter.data
            elif con_filter is not None:  # must be int or tuple of int
                data = uniform_filter(ds, con_filter).data
            else:
                data = ds.data
            ax.contour(x, ds.t, data, levels=levels,
                       linestyles='solid', colors='k', linewidths=0.5)
        ph.lbl_map(ax, symlog)

    def spec(self, t_list, norm=False, ax=None, n_average=0, **kwargs):
        """
        Plot spectra at given times.

        Parameters
        ----------
        t_list : list of floats
            List of the times where the spectra are plotted.
        norm : bool
            If true, each spectral will be normalized.
        ax : plt.Axis or None.
            Axis where the spectra are plotted. If none, the current axis will
            be used.
        n_average : int
            For noisy data it may be prefered to average multiple spectra
            together. This function plots the average of `n_average` spectra
            around the specific time-points.

        Returns
        -------
        list of `Lines2D`
            List containing the Line2D objects belonging to the spectra.
        """

        if ax is None:
            ax = plt.gca()
        is_nm = self.freq_unit == 'nm'
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        ds = self.dataset
        x = ds.wavelengths if is_nm else ds.wavenumbers
        li = []
        for i in t_list:
            idx = dv.fi(ds.t, i)
            if n_average > 0:
                dat = uniform_filter(ds, (2 * n_average + 1, 1))[idx, :]
            elif n_average == 0:
                dat = ds.data[idx, :]
            else:
                raise ValueError(
                    'n_average must be an Integer greater or equal 0.')

            if norm:
                dat = dat / abs(dat).max()
            li += ax.plot(x, dat,
                          label=ph.time_formatter(ds.t[idx], ph.time_unit),
                          **kwargs)

        ax.set_xlabel(ph.freq_label)
        ax.set_ylabel(ph.sig_label)
        ax.autoscale(1, 'x', 1)
        ax.axhline(0, color='k', lw=0.5, zorder=1.9)
        ax.legend(loc='best', ncol=2, title='Delay time')
        return li

    def trans(self, wls, symlog=True, norm=False, ax=None,
              **kwargs):
        """
        Plot the nearest transients for given frequencies.

        Parameters
        ----------
        wls : list of float
            Spectral positions, should be given in the same unit as
            `self.freq_unit`.
        symlog : bool
            Determines if the x-scale is symlog.
        norm : bool or float
            If `False`, no normalization is used. If `True`, each transient
            is divided by the maximum absolute value. If `norm` is a float,
            all transient are normalized by their signal at the time `norm`.
        ax : plt.Axes or None
            Takes a matplotlib axes. If none, it uses `plt.gca()` to get the
            current axes. The lines are plotted in this axis.

        All other kwargs are forwarded to the plot function.

        Returns
        -------
         list of Line2D
            List containing the plotted lines.
        """
        if ax is None:
            ax = plt.gca()
        ds = self.dataset
        wl, t, d = ds.wl, ds.t, ds.data
        l, plotted_vals = [], []
        for i in wls:
            if self.freq_unit is 'nm':
                idx = dv.fi(wl, i)
            else:
                idx = dv.fi(ds.wavenumbers, i)
            dat = d[:, idx]
            if norm is True:
                dat = np.sign(dat[np.argmax(abs(dat))]) * dat / abs(dat).max()
            elif norm is False:
                pass
            else:
                dat = dat / dat[dv.fi(t, norm)]
            plotted_vals.append(dat)
            l.extend(ax.plot(t, dat, label='%.1f %s' % (wl[idx], ph.freq_unit),
                             **kwargs))

        if symlog:
            ax.set_xscale('symlog', linthreshx=1.)
        ph.lbl_trans(ax=ax, use_symlog=symlog)
        ax.legend(loc='best', ncol=2, title='Wavelength')
        ax.set_xlim(right=t.max())
        return l

    def overview(self):
        """
        Plots an overview figure.
        """
        is_nm = self.freq_unit is 'nm'
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        ds = self.dataset
        x = ds.wavelengths if is_nm else ds.wavenumbers
        fig, axs = plt.subplots(3, 1, figsize=(5, 12),
                                gridspec_kw=dict(height_ratios=(2, 1, 1)))
        self.map(ax=axs[0])

        times = np.hstack((0, np.geomspace(0.1, ds.t.max(), 6)))
        sp = self.spec(times, ax=axs[1])
        freqs = np.unique(np.linspace(x.min(), x.max(), 6))
        tr = self.trans(freqs, ax=axs[2])
        OverviewPlot = namedtuple('OverviewPlot', 'fig axs trans spec')
        return OverviewPlot(fig, axs, tr, sp)

    def svd(self, n=5):
        """
        Plot the SVD-components of the dataset.

        Parameters
        ----------
        n : int or list of int
            Determines the plotted SVD-components. If `n` is an int, it plots
            the first n components. If `n` is a list of ints, then every
            number is a SVD-component to be plotted.
        """
        is_nm = self.freq_unit is 'nm'
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        ds = self.dataset
        x = ds.wavelengths if is_nm else ds.wavenumbers
        fig, axs = plt.subplots(3, 1, figsize=(4, 5))
        u, s, v = np.linalg.svd(ds.data)
        axs[0].stem(s)
        axs[0].set_xlim(0, 11)
        try:
            len(n)
            comps = n
        except TypeError:
            comps = range(n)

        for i in comps:
            axs[1].plot(ds.t, u.T[i], label=f"{i}")
            axs[2].plot(x, v[i])
        ph.lbl_trans(axs[1], use_symlog=True)
        ph.lbl_spec(axs[2])


class DataSetInteractiveViewer:
    def __init__(self, dataset, fig_kws={}):
        """
        Class showing a interactive matplotlib window for exploring
        a dataset.
        """
        import matplotlib.pyplot as plt
        self.dataset = dataset
        self.figure, axs = plt.subplots(3, 1, **fig_kws)
        self.ax_img, self.ax_trans, self.ax_spec = axs
        self.ax_img.pcolormesh(dataset.wl, dataset.t, dataset.data)
        self.ax_img.set_yscale('symlog', linscaley=1)

        self.trans_line = self.ax_trans.plot()
        self.spec_line = self.ax_spec.plot()

    def init_event(self):
        'Connect mpl events'
        connect = self.figure.canvas.mpl_connect
        connect('motion_notify_event', self.update_lines)

    def update_lines(self, event):
        """If the mouse cursor is over the 2D image, update
        the dynamic transient and spectrum"""
        pass
