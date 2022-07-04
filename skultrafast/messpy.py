from dis import dis
import numpy as np
import lmfit
from pathlib import Path
import os
import attr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import trim_mean
from typing import Callable, Literal, Tuple, Union, Optional, no_type_check, Dict
import h5py

from skultrafast.dataset import TimeResSpec, PlotterMixin, PolTRSpec
from skultrafast.twoD_dataset import TwoDim
from skultrafast import plot_helpers as ph
from skultrafast.utils import sigma_clip, gauss_step, poly_bg_correction
from skultrafast.dv import make_fi, subtract_background
from skultrafast.unit_conversions import THz2cm


class MessPy2File:
    """
    Class for working with older messpy2 files.
    """

    def __init__(self, fname: os.PathLike):
        self.file = np.load(fname,  allow_pickle=True)

    def get_meta_info(self):
        meta = self.file['meta']
        meta.shape = (1, )
        samp = meta[0]['children']['Sample']['children']
        out = {}
        for i in samp.keys():
            out[i] = samp[i]['value']
        return out

    def vis_wls(self, slope=-1.5, intercept=864.4):
        return slope * np.arange(390) + intercept

    def process_vis(self, vis_range=(390, 720), min_scan=None,
                    max_scan=None, sigma=2.3, para_angle=45):
        data_file = self.file
        wls = self.vis_wls()
        t = data_file['t']
        d = -data_file['data_Stresing CCD'][min_scan:max_scan, ...]

        if 'rot' in data_file:
            rot = data_file['rot'][min_scan:max_scan]
            para_idx = (abs(np.round(rot) - para_angle) < 3)
        else:
            n_ir_cwl = data_file['wl_Remote IR 32x2'].shape[0]
            para_idx = np.repeat(np.array([False, True], dtype='bool'), n_ir_cwl)

        dpm = sigma_clip(d[para_idx, ...], axis=0, sigma=sigma, max_iter=10)
        dsm = sigma_clip(d[~para_idx, ...], axis=0, sigma=sigma, max_iter=10)
        dp = dpm.mean(0)
        dps = dpm.std(0)
        ds = dsm.mean(0)
        dss = dsm.std(0)

        para = TimeResSpec(wls, t, dp[0, :, 0, ...], freq_unit='nm',
                           disp_freq_unit='nm')
        perp = TimeResSpec(wls, t, ds[0, :, 0, ...], freq_unit='nm',
                           disp_freq_unit='nm')
        pol = PolTRSpec(para, perp)

        pol = pol.cut_freq(*vis_range, invert_sel=True)
        return pol.para, pol.perp, pol

    def process_ir(self, t0=0, min_scans=0,
                   max_scans=None, subtract_background=True,
                   center_ch=16, disp=14, sigma=3) -> PolTRSpec:
        data_file = self.file
        t = data_file['t'] - t0
        wli = data_file['wl_Remote IR 32x2']
        print(wli[:, 16, None])

        wli = -(disp * (np.arange(32) - center_ch)) + wli[:, 16, None]
        wli = 1e7 / wli
        d = data_file['data_Remote IR 32x2'][min_scans:max_scans]
        print(d.shape)
        dp = sigma_clip(d[1::2, ...], axis=0, sigma=sigma)
        dpm = dp.mean(0)
        ds = sigma_clip(d[0::2, ...], axis=0, sigma=sigma)
        dsm = ds.mean(0)

        if subtract_background:
            dsm -= dsm[:, :10, ...].mean(1, keepdims=True)
            dpm -= dpm[:, :10, ...].mean(1, keepdims=True)

        para = TimeResSpec(wli[0],
                           t,
                           dpm[0, :, 0, ...],
                           freq_unit='cm',
                           disp_freq_unit='cm')
        perp = TimeResSpec(wli[0],
                           t,
                           dsm[0, :, 0, ...],
                           freq_unit='cm',
                           disp_freq_unit='cm')

        para.plot.spec(1, n_average=20)

        for i in range(1, wli.shape[0]):
            para_t = TimeResSpec(wli[i],
                                 t,
                                 dpm[i, :, 0, ...],
                                 freq_unit='cm',
                                 disp_freq_unit='cm')

            para_t.plot.spec(1, n_average=20)
            para = para.concat_datasets(para_t)

            perp = perp.concat_datasets(
                TimeResSpec(wli[i],
                            t,
                            dsm[i, :, 0, ...],
                            freq_unit='cm',
                            disp_freq_unit='cm'))
        both = PolTRSpec(para, perp)
        return both


class MessPyFile:
    def __init__(
            self,
            fname,
            invert_data=False,
            is_pol_resolved=False,
            pol_first_scan="unknown",
            valid_channel=None,
    ):
        """Class for working with data files from MessPy v1.

        Parameters
        ----------
        fname : str
            Filename to open.
        invert_data : bool (optional)
            If True, invert the sign of the data. `False` by default.
        is_pol_resolved : bool (optional)
            If the dataset was recorded polarization resolved.
        pol_first_scan : {'magic', 'para', 'perp', 'unknown'}
            Polarization between the pump and the probe in the first scan. If
            `valid_channel` is 'both', this corresponds to the zeroth channel.
        valid_channel : `0`, `1`, 'both'
            Indicates which channels contains a real signal. For recently
            recorded data, it is 0 for the visible setup and 1 for the IR
            setup. Older IR data uses both. If `None` it guesses the valid
            channel from the data, assuming recent data.
        """

        with np.load(fname) as f:
            self.wl = f["wl"]
            self.initial_wl = self.wl.copy()
            self.t = f["t"] / 1000.0
            self.data = f["data"]
            if invert_data:
                self.data *= -1

        self.wavenumbers = 1e7 / self.wl
        self.pol_first_scan = pol_first_scan
        self.is_pol_resolved = is_pol_resolved
        if valid_channel is not None:
            self.valid_channel = valid_channel
        else:
            self.valid_channel = 1 if self.wl.shape[0] > 32 else 0

        self.num_cwl = self.data.shape[0]
        self.plot = MessPyPlotter(self)
        self.t_idx = make_fi(self.t)

    def average_scans(self,
                      sigma=3,
                      max_iter=3,
                      min_scan=0,
                      max_scan=None,
                      disp_freq_unit=None):
        """
        Calculate the average of the scans. Uses sigma clipping, which
        also filters nans. For polarization resolved measurements, the
        function assumes that the polarisation switches every scan.

        Parameters
        ----------
        sigma : float
            sigma used for sigma clipping.
        max_iter: int
            Maximum iterations in sigma clipping.
        min_scan : int or None
            All scans before min_scan are ignored.
        max_scan : int or None
            If `None`, use all scan, else just use the scans up to max_scan.
        disp_freq_unit : 'nm', 'cm' or None
            Sets `disp_freq_unit` of the created datasets.

        Returns
        -------
        dict or TimeResSpec
            TimeResSpec or Dict of DataSets containing the averaged datasets. If
            the first delay-time are identical, they are interpreted as
            background and their mean is subtracted.

        """
        if max_scan is None:
            sub_data = self.data
        else:
            sub_data = self.data[..., min_scan:max_scan]
        num_wls = self.data.shape[0]
        t = self.t
        if disp_freq_unit is None:
            disp_freq_unit = "nm" if self.wl.shape[0] > 32 else "cm"
        kwargs = dict(disp_freq_unit=disp_freq_unit)

        if not self.is_pol_resolved:
            data = sigma_clip(sub_data, sigma=sigma, max_iter=max_iter, axis=-1)
            mean = data.mean(-1)
            std = data.std(-1)
            err = std / np.sqrt((~data.mask).sum(-1))

            if self.valid_channel in [0, 1]:
                mean = mean[..., self.valid_channel]
                std = std[..., self.valid_channel]
                err = err[..., self.valid_channel]

                out = {}

                if num_wls > 1:
                    for i in range(num_wls):
                        ds = TimeResSpec(self.wl[:, i], t, mean[i, ..., :], err[i, ...],
                                         **kwargs)
                        out[self.pol_first_scan + str(i)] = ds
                else:
                    out = TimeResSpec(self.wl[:, 0], t, mean[0, ...], err[0, ...],
                                      **kwargs)
                return out
            else:
                raise NotImplementedError("TODO")

        elif self.is_pol_resolved and self.valid_channel in [0, 1]:
            assert self.pol_first_scan in ["para", "perp"]
            data1 = sigma_clip(sub_data[..., self.valid_channel, ::2],
                               sigma=sigma,
                               max_iter=max_iter,
                               axis=-1)
            mean1 = data1.mean(-1)
            std1 = data1.std(-1, ddof=1)
            err1 = std1 / np.sqrt(np.ma.count(data1, -1))

            data2 = sigma_clip(sub_data[..., self.valid_channel, 1::2],
                               sigma=sigma,
                               max_iter=max_iter,
                               axis=-1)
            mean2 = data2.mean(-1)
            std2 = data2.std(-1, ddof=1)
            err2 = std2 / np.sqrt(np.ma.count(data2, -1))

            out = {}
            for i in range(num_wls):
                wl, t = self.wl[:, i], self.t
                if self.pol_first_scan == "para":
                    para = mean1[i, ...]
                    para_err = err1[i, ...]
                    perp = mean2[i, ...]
                    perp_err = err2[i, ...]
                elif self.pol_first_scan == "perp":
                    para = mean2[i, ...]
                    para_err = err2[i, ...]
                    perp = mean1[i, ...]
                    perp_err = err1[i, ...]

                para_ds = TimeResSpec(wl, t, para, para_err, **kwargs)
                perp_ds = TimeResSpec(wl, t, perp, perp_err, **kwargs)
                out["para" + str(i)] = para_ds
                out["perp" + str(i)] = perp_ds
                iso = 1/3*para + 2/3*perp
                out["iso" + str(i)] = TimeResSpec(wl, t, iso, **kwargs)
            self.av_scans_ = out
            return out
        else:
            raise NotImplementedError("Iso correction not supported yet.")

    def recalculate_wavelengths(self, dispersion, center_ch=None, offset=0):
        """Recalculates the wavelengths, assuming linear dispersion.
        Currently assumes that the wavelength set by spectrometer is stored
        in channel 16

        Parameters
        ----------
        dispersion : float
            The dispersion per channel.
        center_ch : int
            Determines the mid-channel. Defaults to len(wl)/2.
        """
        n = self.wl.shape[0]
        if center_ch is None:
            center_ch = n // 2

        # Here we assume that the set wavelength of the spectrometer
        # is written in channel 16
        center_wls = self.initial_wl[16, :]
        new_wl = (np.arange(-n // 2, n // 2) + (center_ch-16)) * dispersion
        self.wl = np.add.outer(new_wl, center_wls) + offset
        self.wavenumbers = 1e7 / self.wl
        if hasattr(self, "av_scans_"):
            for k, i in self.av_scans_.items():
                idx = int(k.strip("paraisoperp"))
                i.wavenumbers = self.wavenumbers[:, idx]
                i.wavelengths = self.wl[:, idx]

    def subtract_background(self, n=10, prop_cut=0):
        """Substracts the the first n-points of the data"""
        back = trim_mean(self.data[:, :n, ...], prop_cut, axis=0)
        self.data -= back

    def avg_and_concat(self):
        """Averages the data and concatenates the resulting TimeResSpec"""
        if not hasattr(self, "av_scans_"):
            self.average_scans()
        out = []
        for pol in ["para", "perp", "iso"]:
            tmp = self.av_scans_[pol + "0"]
            for i in range(1, self.wl.shape[1]):
                tmp = tmp.concat_datasets(self.av_scans_[pol + str(i)])
            out.append(tmp)
        para, perp, iso = out
        return para, perp, iso


class MessPyPlotter(PlotterMixin):
    def __init__(self, messpyds):
        """
        Class to plot utility plots

        Parameters
        ----------
        messpyds : MessPyFile
            The MessPyDataSet.
        """

        self.ds = messpyds

    def background(self, n=10, ax=None):
        """
        Plot the backgrounds each center wl.
        """
        if ax is None:
            ax = plt.gca()

        out = self.ds.average_scans()
        if isinstance(out, dict):
            for i in out:
                ds = out[i]
                ax.plot(ds.wavelengths, ds.data[:n, :].mean(0), label=i)
            self.lbl_spec(ax)
        else:
            return

    def early_region(self):
        """
        Plots an imshow of the early region.
        """
        n = self.ds.num_cwl
        ds = self.ds
        fig, axs = plt.subplots(1,
                                n,
                                figsize=(n*2.5 + 0.5, 2.5),
                                sharex=True,
                                sharey=True)

        if not hasattr(ds, "av_scans_"):
            return
        for i in range(n):
            d = ds.av_scans_["para" + str(i)].data
            axs[i].imshow(d, aspect="auto")
            axs[i].set_ylim(0, 50)

    def compare_spec(self, t_region=(0, 4), every_nth=1, ax=None):
        """
        Plots the averaged spectra of every central wavelenth and polarisation.

        Parameters
        ----------
        t_region : tuple of floats
            Contains the the start and end-point of the times to average.
        every_nth: int
            Only every n-th scan is plotted.

        Returns
        -------
        None
        """

        if ax is None:
            ax = plt.gca()

        n = self.ds.num_cwl
        ds = self.ds
        t = self.ds.t
        if not hasattr(ds, "av_scans_"):
            self.ds.average_scans()
        for i in range(0, n, every_nth):
            c = "C%d" % i
            sl = (t_region[0] < t) & (t < t_region[1])
            if "para" + str(i) in ds.av_scans_:
                d = ds.av_scans_["para" + str(i)]
                ax.plot(d.wavelengths, d.data[sl, :].mean(0), c=c, lw=2)
            if "perp" + str(i) in ds.av_scans_:
                d = ds.av_scans_["perp" + str(i)]
                ax.plot(d.wavelengths, d.data[sl, :].mean(0), c=c, lw=1)
            elif "iso" + str(i) in ds.av_scans_:
                d = ds.av_scans_["iso" + str(i)]
                ax.plot(d.wavelengths, d.data[sl, :].mean(0), c=c, lw=1)
        ph.lbl_spec(ax)

    def compare_scans(self,
                      t_region=(0, 4),
                      channel=None,
                      cmap="jet",
                      ax=None,
                      every_nth=1):
        """
        Plots the spectrum averaged over `t_region` for `every_nth` scan.
        """
        if ax is None:
            ax = plt.gca()
        if channel is None:
            channel = self.ds.valid_channel
        n_scans = self.ds.data.shape[-1]
        d = self.ds.data
        t = self.ds.t
        sl = (t_region[0] < t) & (t < t_region[1])
        colors = plt.cm.get_cmap(cmap)
        if not self.ds.is_pol_resolved:
            for i in range(0, n_scans, every_nth):
                c = colors(i * every_nth / n_scans)
                for j in range(d.shape[0]):

                    ax.plot(self.ds.wavenumbers[:, j],
                            d[j, sl, :, channel, i].mean(0),
                            label='%d' % i,
                            c=c)
        else:
            for i in range(0, n_scans, 2 * every_nth):
                c = colors(2 * every_nth * i / n_scans)
                for j in range(d.shape[0]):
                    x = self.ds.wavenumbers[:, j]
                    y = d[j, sl, :, channel, i].mean(0)
                    ax.plot(x, y, label="%d" % i, c=c)
                    if i + 1 < n_scans:
                        y = d[j, sl, :, channel, i + 1].mean(0)
                        ax.plot(x, y, label="%d" % (i+1), c=c)

        ph.lbl_spec(ax)


@attr.define
class Messpy25File:
    h5_file: h5py.File = attr.ib(init=True)
    'h5py file object containing the 2D dataset, the only required parameter'
    is_para_array: Literal['Probe1', 'Probe2'] = 'Probe1'
    'which dataset has parallel polarisation'
    probe_wn: np.ndarray = attr.ib(init=False)
    'Array with probe wavenumbers'
    pump_wn: np.ndarray = attr.ib(init=False)
    'Array with the pump wavenumbers. Depends on the upsampling used during measurment'
    t2: np.ndarray = attr.ib(init=False)
    'Array containing the waiting times'
    t1: np.ndarray = attr.ib(init=False)
    'Array containing the double pulse delays'
    rot_frame: float = attr.ib(init=False)
    'Rotation frame used while measuring'

    @no_type_check
    def __attrs_post_init__(self):
        if 't1' in self.h5_file:
            self.t1 = self.h5_file['t1'][:]
            self.t2 = self.h5_file['t2'][:]
        else:
            self.t1 = self.h5_file['t2'][:]
            self.t2 = self.h5_file['t3'][:]
        self.rot_frame = self.h5_file['t1'].attrs['rot_frame']
        self.probe_wn = self.h5_file['wn'][:]
        i: np.ndarray = self.h5_file['ifr_data/Probe1/0/0']
        self.pump_wn = THz2cm(np.fft.rfftfreq(2*i.shape[1], (self.t1[1]-self.t1[0])))
        self.pump_wn += self.rot_frame

    def get_means(self):
        means = {}
        for name, l in self.h5_file['2d_data'].items():
            means[name] = []
            for i in range(self.t2.size):
                means[name].append(l[str(i)]['mean'])
        para = self.is_para_array

        perp = "Probe2" if self.is_para_array == "Probe1" else "Probe2"
        para_means = np.stack(means[para], 0)
        perp_means = np.stack(means[perp], 0)
        return para_means, perp_means, 2/3*perp_means + 1/3*para_means

    def get_ifr(self, probe_filter=None, bg_correct=None, ch_shift: int = 0):
        """
        Returns the interferograms. If necessary, apply probefilter and background correction.

        Parameters
        ----------
        probe_filter: float
            The probe filter width in channels. (Gaussian filter)
        bg_correct: Tuple[int, int]
            Number of left and right channels to use for background correction.
        ch_shift: int
            Number of channels to shift the Probe2 data. Corrects for missaligned channels.
        Returns
        -------
        ifr: Tuple[np.ndarray, np.ndarray, np.ndarray]
            The interferograms for paralllel, perpendicular and isotropic polarisation.
            The shape of each array is (n_t2, n_probe_wn, n_t1).
        """
        ifr = {}
        for name, l in self.h5_file['ifr_data'].items():
            ifr[name] = []
            for i in range(self.t2.size):
                ifr[name].append(l[str(i)]['mean'])
        para = self.is_para_array
        perp = "Probe2" if self.is_para_array == "Probe1" else "Probe1"

        para_means = np.stack(ifr[para], 0)
        perp_means = np.stack(ifr[perp], 0)
        if probe_filter is not None:
            para_means = gaussian_filter1d(
                para_means, probe_filter, 1, mode='nearest')
            perp_means = gaussian_filter1d(
                perp_means, probe_filter, 1, mode='nearest')
        if ch_shift > 0:
            para_means = para_means[:, :-ch_shift, :]
            perp_means = perp_means[:, ch_shift:, :]
        elif ch_shift < 0:
            para_means = para_means[:, -ch_shift:, :]
            perp_means = perp_means[:, :-ch_shift, :]

        if bg_correct is not None:
            for i in range(para_means.shape[0]):
                poly_bg_correction(
                    self.probe_wn[ch_shift:], para_means[i].T, bg_correct[0], bg_correct[1])
                poly_bg_correction(
                    self.probe_wn[ch_shift:], perp_means[i].T, bg_correct[0], bg_correct[1])

        return para_means, perp_means, 2/3*perp_means + 1/3*para_means

    def make_two_d(self, upsample: int = 4, window_fcn: Optional[Callable] = np.hanning, ch_shift: int = 1,
                   probe_filter: Optional[float] = None, bg_correct: Optional[Tuple[int, int]] = None) -> Dict[str, TwoDim]:
        """
        Calculates the 2D spectra from the interferograms and returns it as a dictionary.
        The dictorary contains messpy 2D-objects for paralllel, perpendicular and isotropic polarisation.

        Parameters
        ----------
        upsample: int
            Upsampling factor used in the FFT. A factor over 2 only does sinc interpolation.
        window_fcn: Callable
            If given, apply a window function to the FFT.
        probe_filter: float
            The probe filter width in channels. (Gaussian filter)
        ch_shift: int
            Number of channels to shift the Probe2 data. Corrects for missaligned channels.
        bg_correct: Tuple[int, int]
            Number of left and right channels to use for background correction.
        """
        means = self.get_ifr(probe_filter=probe_filter,
                             bg_correct=bg_correct, ch_shift=ch_shift)
        data = {pol: means[i] for i, pol in enumerate(['para', 'perp', 'iso'])}
        out = {}
        for k, v in data.items():
            v[:, :, 0] *= 0.5
            if window_fcn is not None:
                v = v*window_fcn(v.shape[2]*2)[None, None, v.shape[2]:]
            sig = np.fft.rfft(v, axis=2, n=v.shape[2]*upsample).real
            self.pump_wn = THz2cm(np.fft.rfftfreq(
                upsample*v.shape[2], (self.t1[1]-self.t1[0]))) + self.rot_frame
            ds = TwoDim(self.t2, self.pump_wn, self.probe_wn[ch_shift:], sig)
            out[k] = ds
        return out

    def make_model_fitfiles(self, path, name, probe_filter=None, bg_correct=None):
        """
        Saves the data in a format useable for the ModelFit Gui from Kevin Robben
        https://github.com/kevin-robben/model-fitting
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        ifr = self.get_ifr(probe_filter=probe_filter, bg_correct=bg_correct)
        data = {pol: ifr[i] for i, pol in enumerate(['para', 'perp', 'iso'])}
        idx = np.argsort(self.probe_wn)

        for pol in ['para', 'perp', 'iso']:
            folder = p / pol
            folder.mkdir(parents=True, exist_ok=True)
            for i, t in enumerate(self.t2):
                fname = folder / (name + '_%f.txt' % t)
                d = data[pol][i, idx, :]

                np.savetxt(fname, d)
        np.savetxt(p / 'pump_wn.txt', self.pump_wn)
        np.savetxt(p / 'probe_wn.calib', self.probe_wn[idx])
        np.savetxt(p / 't1.txt', self.t1)
        np.savetxt(p / 't2.txt', self.t2)
        timestep = (self.t1[1] - self.t1[0])*1000

        np.savetxt(
            p / f"rot_frame_{self.rot_frame: .0f}_t1_stepfs_{timestep: .0f}.txt", [self.rot_frame])

    def recalculate_wl(self, center_wl=None, center_ch: int = 65, disp: Optional[float] = None):
        """
        Recalculates the wavelengths from the probe.
        """
        if disp is None:
            if np.diff(1e7/self.probe_wn).max() < 6:
                disp = 7.8/2
            else:
                disp = 7.8
        wls = center_wl - disp*(np.arange(128)-center_ch)
        self.probe_wn = 1e7/wls


@attr.s(auto_attribs=True)
class TzResult:
    x0: float
    sigma: float
    fwhm: float
    fit_result: lmfit.model.ModelResult
    data: Tuple[np.ndarray, np.ndarray]
    fig: Optional[plt.Figure] = None


def get_t0(fname: str,
           sigma: float = 1,
           scan: Union[int, slice] = -1,
           display_result: bool = True,
           plot: bool = True,
           t_range: Tuple[float, float] = (-2, 2),
           invert: bool = False,
           no_slope: bool = True) -> TzResult:
    """Determine t0 from a semiconductor messuarement in the IR. For that, it opens
    the given file, takes the mean of all channels and fits the resulting curve with
    a step function.

    Note that the parameter

    Parameters
    ----------
    fname : str
        Filename of the messpy file containing the data
    sigma : float, optional
        Used for calculating the displayed nummerical derviate, by default 1.
    scan: int or slice
        Which scan to use, by default -1, the last scan. If given a slice,
        it takes the mean of the scan slice.
    display_result : bool, optional
        If true, show the fitting results, by default True
    plot : bool, optional
        If true, plot the result, by default True
    t_range : (float, flot)
        The range which is used to fit the data.
    invert : bool
        If true, invert data.
    no_slope : bool
        Determines if a variable slope is added to the fit model.

    Returns
    -------
    TzResult
        Result and presentation of the fit.
    """
    a = np.load(fname, allow_pickle=True)
    if not fname[-11:] == 'messpy1.npz':
        if 'data' in a:
            data = a['data']
            if isinstance(scan, slice):
                sig = np.nanmean(data[0, ..., scan], axis=-1)
            else:
                sig = data[0, ..., scan]
            sig = np.nanmean(sig[:, :, 1], axis=1)
        else:
            sig = np.nanmean(a['signal'], 1)

        t = a['t'] / 1000.
    else:
        data = a['data_Remote IR 32x2']
        if isinstance(scan, slice):
            sig = np.nanmean(data[scan, ...], axis=0)
        else:
            sig = data[scan, ...]
        sig = np.nanmean(sig[0, :, 1, :], axis=-1)
        t = a['t']
    if invert:
        sig = -sig

    idx = (t > t_range[0]) & (t < t_range[1])
    sig = sig.squeeze()[idx]
    #from scipy.signal import savgol_filter
    #dsig = savgol_filter(sig, 11, 2, 1)
    dsig = gaussian_filter1d(sig, sigma=1, order=1)

    GaussStep = lmfit.Model(gauss_step)

    model = GaussStep + lmfit.models.LinearModel()
    max_diff_idx = np.argmax(abs(dsig))

    params = model.make_params(amp=np.ptp(sig),
                               center=t[idx][max_diff_idx],
                               sigma=0.2,
                               slope=0,
                               intercept=sig.min())
    if no_slope:
        params['slope'].vary = False
    params.add(lmfit.Parameter('FWHM', expr="sigma*2.355"))
    result = model.fit(params=params, data=sig, x=t[idx])
    fig = None
    if display_result:
        import IPython.display
        IPython.display.display(result.params)
    if plot:
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(5, 7),
        )

        axs[0].plot(t[idx], sig)
        tw = axs[0].twinx()
        tw.plot(t[idx], dsig, c='r', label='Nummeric Diff')
        tw.legend()
        #axs[1].plot(t[idx], dsig, color='red')
        axs[1].set_xlabel('t')
        plt.sca(axs[1])
        result.plot_fit()
        axs[1].axvline(result.params['center'].value)

    res = TzResult(
        x0=result.params['center'],
        sigma=result.params['sigma'],
        fwhm=result.params['FWHM'],
        data=(t[idx], sig),
        fit_result=result,
    )
    return res
