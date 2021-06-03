from typing import Dict
import numpy as np
from pathlib import Path
import attr

from skultrafast import dv
from skultrafast.dataset import TimeResSpec


def inbetween(a, lower, upper):
    return np.logical_and(a >= lower, a <= upper)


@attr.s(auto_attribs=True)
class TwoDim:
    t: np.ndarray
    "Array of the waiting times"
    pump_wn: np.ndarray
    "Array with the pump-wavenumbers"
    probe_wn: np.ndarray
    "Array with the probe-wavenumbers"
    data: np.ndarray
    "Array with the data"
    info: Dict = {}
    "Meta Info"
    plot: 'TwoDimPlotter' = attr.ib()

    @plot.default
    def _plot_default(self):
        return TwoDimPlotter(self)

    def __attrs_post_init__(self):
        n, m, k = self.t.size, self.probe_wn.size, self.pump_wn.size
        if self.data.shape != (n, m, k):
            raise ValueError("Data shape not equal to t, wn_probe, wn_pump shape"
                             f"{self.data.shape} != {n, m, k}")

        self.data = self.data.copy()
        self.probe_wn = self.data.copy()
        self.probe_wn = self.data.copy()

        i1 = np.argsort(self.pump_wn)
        i2 = np.argsort(self.probe_wn)

    def copy(self) -> 'TwoDim':
        return attr.evolve(self)

    def t_idx(self, t: float) -> int:
        "Return nearest idx to nearest time value"
        return dv.fi(self.t, t)

    def probe_idx(self, wn: float) -> int:
        "Return nearest idx to nearest probe_wn value"
        return dv.fi(self.probe_wn, wn)

    def pump_idx(self, wn: float) -> int:
        "Return nearest idx to nearest pump_wn value"
        return dv.fi(self.pump_wn, wn)

    def select_pump(self, pump_range, probe_range, invert=False) -> 'TwoDim':
        """
        Return a dataset containing only the selected region.               
        """
        pu_idx = inbetween(self.pump_wn, min(pump_range), max(pump_range))
        pr_idx = inbetween(self.probe_wn, min(probe_range), max(probe_range))

        if invert:
            pu_idx = not pu_idx
            pr_idx = not pr_idx

        ds = self.copy()
        ds.data = ds.data[:, pr_idx, :][:, :, pu_idx]
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
        data = self.data[:, :, pu_idx].sum(-1)
        return TimeResSpec(self.pump_wn, self.t, data, freq_unit='cm')

    def cls(self):
        pass


@attr.s(auto_attribs=True)
class TwoDimPlotter:
    ds: TwoDim

    def contour(self, *times):
        pass
