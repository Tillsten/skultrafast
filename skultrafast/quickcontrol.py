"""
Module to import and work with files generated by QuickControl from phasetech.
"""
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sized, Tuple, Union

import attr
import numpy as np
from scipy.constants import speed_of_light
from scipy.ndimage import gaussian_filter1d

from skultrafast.dataset import PolTRSpec, TimeResSpec
from skultrafast.twoD_dataset import TwoDim
from skultrafast.utils import poly_bg_correction


def parse_str(s: str):
    """
    Parse entry of info file

    Parameters
    ----------
    s : str
        Value
    Returns
    -------
    obj
        Corresponding python type
    """
    if s.isnumeric():
        return int(s)
    elif set(s) - set('-.0123456789E') == set():
        # Try is a workaround for the version string
        try:
            return float(s)
        except ValueError:
            return s
    elif set(s) - set('-.0123456789E,') == set():
        return list(map(float, s.split(',')))
    elif s == 'TRUE':
        return True
    elif s == 'FALSE':
        return False
    else:
        return s


@attr.s(auto_attribs=True)
class QCFile:
    """
    Base class for QC files.
    """
    fname: str = attr.ib()
    """Full path to info file"""

    path: Path = attr.ib()
    """Directory of the info file"""

    prefix: str = attr.ib()
    """Filename"""

    info: dict = attr.ib()
    """Content of the info file"""
    @path.default
    def _path(self):
        return Path(self.fname).parent

    @prefix.default
    def _prefix(self):
        return Path(self.fname).with_suffix('').name

    @info.default
    def _load_info(self):
        h = []
        d = {}
        with (self.path / self.prefix).with_suffix('.info').open() as i:
            for l in i:
                key, val = l.split('\t')
                val = val[:-1].strip()
                d[key] = parse_str(val)
        return d


@attr.s(auto_attribs=True)
class QCBaserTimeRes(QCFile):
    wavelength: np.ndarray = attr.ib()
    """Wavelength data calculated from given grating and mono wavelength"""

    @wavelength.default
    def calc_wl(self, disp=None):
        if disp is None:
            grating = self.info['MONO1 Grating']
            disp_per_grating = {'30': 7.7, '75': 7.7 * 30 / 75.}
            disp = disp_per_grating[grating.split()[2]]
        wls = (np.arange(128) - 64) * disp + self.info['MONO1 Wavelength']
        self.wavelength = wls
        return wls

    @property
    def wavenumbers(self):
        return 1e7 / self.wavelength


@attr.s(auto_attribs=True)
class QC1DSpec(QCBaserTimeRes):
    """Helper class to load time resolved spectra measured with QuickControl"""

    t: Iterable[float] = attr.ib()
    """Delay times """

    par_data: np.ndarray = attr.ib()
    """"Contains the data from one channel"""

    per_data: np.ndarray = attr.ib()
    """"Contains the data from one channel"""

    @par_data.default
    def _load_par(self):
        par_scan_files = self.path.glob(self.prefix + '*_PAR*.scan')
        return np.array([np.loadtxt(p)[:-1, 1:] for p in par_scan_files])

    @per_data.default
    def _load_per(self):
        per_scan_files = self.path.glob(self.prefix + '*_PER*.scan')
        return np.array([np.loadtxt(p)[:-1, 1:] for p in per_scan_files])

    @t.default
    def _t_default(self):
        t_list = np.array(self.info['Delays'])
        if self.info['Delay Units'] == 'fs':
            t_list /= 1000.
        return t_list

    def make_pol_ds(self, sigma=None) -> PolTRSpec:
        para = np.nanmean(self.par_data, axis=0)
        ds_para = TimeResSpec(self.wavelength, self.t, 1000 * para, disp_freq_unit='cm')
        perp = np.nanmean(self.per_data, axis=0)
        ds_perp = TimeResSpec(self.wavelength, self.t, 1000 * perp, disp_freq_unit='cm')
        return PolTRSpec(ds_para, ds_perp)


@attr.s(auto_attribs=True)
class QC2DSpec(QCBaserTimeRes):
    """Helper class to load 2D-spectra measured with QuickControl"""
    t: np.ndarray = attr.ib()
    """Waiting times"""

    t2: np.ndarray = attr.ib()
    """t2, the inter-pulse delays between pump pulses"""

    par_data: Dict = attr.ib()
    """Data for parallel polarization"""

    per_data: Dict = attr.ib()
    """Data for perpendicular polarization"""

    par_spec: Optional[Dict] = None
    """Resulting 2D spectra for parallel polarization"""

    per_spec: Optional[Dict] = None
    """Resulting 2D spectra for perpendicular polarization"""

    probe_filter: Optional[float] = None
    """Size of the filter applied to the spectral axis. 'None' is no filtering"""

    upsampling: int = 2
    """Upsamling factor of the pump-axis"""

    pump_freq: np.ndarray = attr.ib()
    """
    Resulting wavenumbers of the pump axis.
    """

    bg_correct: Optional[Tuple] = None
    """
    If given, the size of the left and right region used to calculate the signal background which
    will be subtracted
    """

    win_function: Optional[Callable] = np.hamming
    """
    Window function used for apodization. The coded will the window-function with
    `2*len(t2)` and uses only the second half of the returned array.
    """

    @t.default
    def _t_default(self):
        t_list = np.array(self.info['Waiting Time Delays'])
        if self.info['Waiting Time Delay Units'] == 'fs':
            t_list /= 1000
        return t_list

    @t2.default
    def _load_t2(self):
        end = self.info['Final Delay (fs)']
        step = self.info['Step Size (fs)']
        return np.arange(0.0, end + 1, step) / 1000.

    def _loader(self, which: str):
        data_dict: Dict[int, np.ndarray] = {}

        for t in range(len(self.t)):
            T = '_T%02d' % (t+1)
            pol_scans = self.path.glob(self.prefix + T + f'_{which}*.scan')
            data = []
            for s in pol_scans:

                d = np.loadtxt(s)
                self.t2 = d[1:, 0]
                data.append(d)
            if len(data) > 0:
                d = np.array(data)
                data_dict[t] = d
            else:
                self.t = self.t[:t + 1]
                break
        return data_dict

    def switch_pol(self):
        self.par_data, self.per_data = self.per_data, self.par_data
        self.par_spec, self.per_spec = self.per_spec, self.par_spec

    def calc_spec(self):
        par_dict: Dict[int, np.ndarray] = {}
        perp_dict: Dict[int, np.ndarray] = {}
        for i, data in enumerate((self.par_data, self.per_data)):
            for t in data:
                d = np.nanmean(data[t], 0)
                d = d[:-1, 1:]

                if self.probe_filter is not None:
                    d = gaussian_filter1d(d, self.probe_filter, -1, mode='nearest')
                if self.bg_correct:
                    poly_bg_correction(self.wavelength, d,
                                       self.bg_correct[0], self.bg_correct[1])
                d[0, :] *= 0.5
                if self.win_function is not None:
                    win = self.win_function(2 * len(self.t2))
                else:
                    win = np.ones(2 * len(self.t2))
                spec = np.fft.rfft(d * win[len(self.t2):, None],
                                   axis=0,
                                   n=self.upsampling * len(self.t2))
                (par_dict, perp_dict)[i][t] = spec.real
        return par_dict, perp_dict

    @par_data.default
    def _load_par(self):
        return self._loader('PAR')

    @per_data.default
    def _load_per(self):
        return self._loader('PER')

    @pump_freq.default
    def _calc_freqs(self):
        freqs = np.fft.rfftfreq(self.upsampling * len(self.t2), self.t2[1] - self.t2[0])
        om0 = self.info['Rotating Frame (Scanned)']
        cm = 10 / ((1/freqs) * 1e-12 * speed_of_light) + om0
        return cm

    def make_ds(self) -> Dict[str, TwoDim]:
        par, perp = self.calc_spec()
        self.pump_freq = self._calc_freqs()
        par_arr = np.dstack(list(par.values())).T
        per_arr = np.dstack(list(perp.values())).T
        iso = (2*per_arr + par_arr) / 3

        d = {
            'para':
            TwoDim(t=self.t,
                   pump_wn=self.pump_freq,
                   probe_wn=self.wavenumbers,
                   spec2d=par_arr),
            'perp':
            TwoDim(t=self.t,
                   pump_wn=self.pump_freq,
                   probe_wn=self.wavenumbers,
                   spec2d=per_arr),
            'iso':
            TwoDim(t=self.t,
                   pump_wn=self.pump_freq,
                   probe_wn=self.wavenumbers,
                   spec2d=iso)
        }
        return d
