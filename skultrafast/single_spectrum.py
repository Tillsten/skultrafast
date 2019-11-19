import numpy as np
from skultrafast import dv, plot_helpers
import matplotlib.pyplot as plt
import lmfit


class SingleSpec:
    def __init__(self, frequency, signal, unit_freq='nm', unit_signal='OD'):
        """
        Class for working with steady state spectra.

        Parameters
        ----------
        frequency : ndarray
            The frequencies of the spectrum
        signal : ndarray
            Array containing the signal.
        unit_freq : str
            Unit type of the frequencies.
        unit_signal : str
            Unit type of the signal.
        """

        assert (frequency.shape[0] == signal.shape[0])
        idx = np.argsort(frequency)
        self.x = frequency[idx]
        self.y = signal[idx, ...]
        self.unit_freq = unit_freq
        self.unit_signal = unit_signal
        self.back = np.zeros_like(self.x)
        self.fi = dv.make_fi(self.x)

    def subtract_const(self, region: tuple):
        """
        Subtracts a constant background. The background is calculated by
        taking the mean signal in the designated region.

        Parameters
        ----------
        region : tuple of two floats
            The borders of the region.
        """
        a, b = region
        i1, i2 = self.fi(a), self.fi(b)
        self.back -= self.y[i1:i2, ...].mean(0)

    def cut(self, region, invert_sel=True):
        """
        Cuts part of the spectrum away.

        Parameters
        ----------
        region : tuple of floats
            Defines the region to be cutted away.
        invert_sel : bool
            If `True`, the cutted region is inverted.

        Returns
        -------
        SingleSpec
            Cut spectra.

        """
        a, b = region
        i1, i2 = self.fi(a), self.fi(b)
        new_x, new_y = self.x[i1:i2], self.y[i1:i2, ...]
        return SingleSpec(new_x, new_y, self.unit_freq, self.unit_signal)

    def fit_single_gauss(self, start_params=None, back_deg=2,
                         peak_region=None):
        peak = lmfit.models.GaussianModel(x=self.x)
        back = lmfit.models.PolynomialModel(degree=back_deg)
        model = peak + back




class SingleSpecPlotter:
    def __init__(self, single_spec: SingleSpec):
        self.ds = single_spec

    def spec(self, remove_back=True, ax=None):
        if ax is None:
            ax = plt.gca()
        ds = self.ds

        ax.plot(ds.x, ds.y - ds.back)
        ax.set_xlabel(ds.unit_freq)
        ax.set_ylabel(ds.unit_signal)
        ax.minorticks_on()
