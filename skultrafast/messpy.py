import numpy as np
from astropy import stats as stats
from skultrafast.dataset import DataSet
import matplotlib.pyplot as plt

def _add_rel_errors():
    pass

class MesspyDataSet:
    def __init__(self, fname, invert_data=False, is_pol_resolved=False,
                 pol_first_scan='unknown', valid_channel='both'):
        """Class for working with data files from MessPy.

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
            setup. Older IR data uses both.
        """

        with np.load(fname) as f:
            self.wl = f['wl']
            self.t = f['t']/1000.
            self.data = f['data']
            if invert_data:
                self.data *= -1

        self.pol_first_scan = pol_first_scan
        self.is_pol_resolved = is_pol_resolved
        self.valid_channel = valid_channel

    def average_scans(self, sigma=3, max_scan=None, disp_freq_unit=None):
        """
        Calculate the average of the scans. Uses sigma clipping, which
        also filters nans. For polarization resolved measurements, the
        function assumes that the polarisation switches every scan.

        Parameters
        ----------
        sigma : float
            sigma used for sigma clipping.
        max_scan : int or None
            If `None`, use all scan, else just use the scans up to max_scan.
        disp_freq_unit : 'nm', 'cm' or None
            Sets `disp_freq_unit` of the created datasets.

        Returns
        -------
        dict or DataSet
            DataSet or Dict of DataSets containing the averaged datasets. If
            the first delay-time are identical, they are interpreted as
            background and their mean is subtracted.

        """
        if max_scan is None:
            sub_data = self.data
        else:
            sub_data = self.data[..., :max_scan]
        num_wls = self.data.shape[0]

        if disp_freq_unit is None:
            disp_freq_unit = 'nm' if self.wl.shape[1] > 32 else 'cm'

        if not self.is_pol_resolved:
            data = stats.sigma_clip(sub_data,
                                    sigma=sigma, axis=-1)
            mean = data.mean(-1)
            std = data.std(-1)
            err = std / np.sqrt((~data.mask).sum(-1))

            if self.valid_channel in [0, 1]:
                mean = data[..., self.valid_channel]
                std = std[..., self.valid_channel]
                err = err[..., self.valid_channel]

                out = {}

                if num_wls > 1:
                    for i in range(num_wls):
                        ds = DataSet(self.wl[:, i], self.t, mean[i, ..., :],
                                     err[i, ...], disp_freq_unit=disp_freq_unit)
                        out[self.pol_first_scan + str(i)] = ds
                else:
                    out = DataSet(self.wl[:, 0], self.t, mean[0, ...],
                                  err[0, ...], disp_freq_unit=disp_freq_unit)
                return out
            else:
                raise NotImplementedError('TODO')

        elif self.is_pol_resolved and self.valid_channel in [0, 1]:
            assert (self.pol_first_scan in ['para', 'perp'])
            data1 = stats.sigma_clip(sub_data[..., self.valid_channel, ::2],
                                     sigma=sigma, axis=-1)
            mean1 = data1.mean(-1)
            std1 = np.ma.std(-1)
            err1 = std1 / np.sqrt(np.ma.count(data1, -1))

            data2 = stats.sigma_clip(sub_data[..., self.valid_channel, 1::2],
                                     sigma=sigma, axis=-1)
            mean2 = data2.mean(-1)
            std2 = data2.std(-1)

            err2 = std2 / np.sqrt(np.ma.count(data2, -1))




            out = {}
            for i in range(num_wls):
                pfs = self.pol_first_scan
                out[pfs + str(i)] = DataSet(self.wl[:, i], self.t,
                                            mean1[i, ...], err1[i, ...],
                                            disp_freq_unit=disp_freq_unit)
                other_pol = 'para' if pfs == 'perp' else 'perp'
                out[other_pol + str(i)] = DataSet(self.wl[:, i], self.t,
                                                  mean2[i, ...], err2[i, ...],
                                                  disp_freq_unit=disp_freq_unit)
                iso = 1/3 * out['para' + str(i)].data + 2 / 3 * out[
                    'perp' + str(i)].data
                iso_err = np.sqrt(
                    1/3 * out['para' + str(i)].err ** 2 + 2 / 3 * out[
                        'perp' + str(i)].data ** 2)
                out['iso' + str(i)] = DataSet(self.wl[:, i], self.t, iso,
                                              iso_err,
                                              disp_freq_unit=disp_freq_unit)
            return out
        else:
            raise NotImplementedError("Iso correction not supported yet.")

    def recalculate_wavelengths(self, dispersion, center_ch=None):
        """Recalculates the wavelengths, assuming linear dispersion.

        Parameters
        ----------
        dispersion : float
            The dispersion per channel.
        center_ch : int
            Determines the mid-channel. Defaults to len(wl)/2.
        """
        n = self.wl.shape[1]
        if center_ch is None:
            center_ch = n//2

        center_wls = self.wl[:, center_ch]
        new_wl = np.arange(-n//2, n//2)*dispersion
        self.wl = np.add.outer(center_wls, new_wl)

class MessPyPlotter:
    def __init__(self, messpyds):
        """
        Class to plot utility plots

        Parameters
        ----------
        messpyds : MesspyDataSet
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
                ax.plot(ds.data[:n, :].mean(0))
        else:
            return
