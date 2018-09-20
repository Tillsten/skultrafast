import numpy as np
import astropy.stats as stats
from skultrafast.data_io import save_txt
from skultrafast.filter import bin_channels
from skultrafast import zero_finding
from enum import Enum
from types import SimpleNamespace
from collections import namedtuple

class Polarization(Enum):
    """Enum which describes the relative polarisation of pump and probe"""
    MAGIC = 'magic'
    PARA = 'para'
    PERP = 'perp'
    CIRC = 'circ'
    UNKNOWN = 'unknown'


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
            Polarization between the pump and the probe in the first scan. 
            If `valid_channel` is 'both', this corresponds to the zeroth channel. 
        valid_channel : {0, 1, 'both'}
            Indicates which channels contains a real signal. 

        """

        with np.load(fname) as f:
            self.wl = f['wl']
            self.t = f['t']
            self.data = f['data']

        self.pol_first_scan = pol_first_scan
        self.is_pol_resolved = is_pol_resolved
        self.valid_channel = valid_channel

    def average_scans(self, sigma=3):
        """Calculate the average of the scans. Uses sigma clipping, which 
        also filters nans. For polarization resovled measurements, the 
        function assumes that the polarisation switches every scan.

        Parameters
        ----------
        sigma : float
           sigma used for sigma clipping.

        Returns
        -------
        : dict or DataSet
            DataSet or Dict of DataSets containing the averaged datasets. 

        """        

        num_wls = self.data.shape[0]

        if not self.is_pol_resolved:
            data = stats.sigma_clip(self.data, 
                                    sigma=sigma, axis=-1)
            data = data.mean(-1)
            std = data.std(-1)
            err = std/np.sqrt(std.mask.sum(-1))
            
            if self.valid_channel in [0, 1]:
                data = data[..., self.valid_channel]
                std = std[..., self.valid_channel]
                err = err[..., self.valid_channel]

                out = {}
                
                if num_wls > 1:
                    for i in range(num_wls):                     
                        ds = DataSet(self.wl[:, i], self.t, data[i, ...], err[i, ...])
                        out[self.pol_first_scan + str(i)] = ds
                else:
                    out = DataSet(self.wl[:, 0], self.t, data[0, ...], err[0, ...])
                return out
            
        elif self.is_pol_resolved and self.valid_channel in [0, 1]:
            assert(self.pol_first_scan in ['para', 'perp'])
            data1 = stats.sigma_clip(self.data[..., self.valid_channel,::2], 
                                    sigma=sigma, axis=-1)
            data1 = data1.mean(-1)
            std1 = data1.std(-1)
            err1 = std1/np.sqrt(data1.mask.sum(-1))
            
            data2 = stats.sigma_clip(self.data[..., self.valid_channel, 1::2], 
                                    sigma=sigma, axis=-1)
            data2 = data2.mean(-1)
            std2 = data2.std(-1)
            err2 = std2/np.sqrt(data2.mask.sum(-1))
        
            out = {}
            for i in range(self.data.shape[0]):
                out[self.pol_first_scan + str(i)] = DataSet(self.wl[:, i], self.t, data1[i, ...], err1[i, ...])
                other_pol = 'para' if self.pol_first_scan == 'perp' else 'perp'
                out[other_pol + str(i)] = DataSet(self.wl[:, i], self.t, data2[i, ...], err2[i, ...])
                iso = 1/3*out['para' + str(i)].data + 2/3*out['perp' + str(i)].data
                iso_err = np.sqrt(1/3*out['para' + str(i)].err**2 + 2/3*out['perp' + str(i)].data**2)
                out['iso' + str(i)] = DataSet(self.wl[:, i], self.t, iso, iso_err)
            return out
        else:
            raise NotImplementedError("Iso correction not suppeorted yet.")


EstDispResult = namedtuple('EstDispResult', 'correct_ds tn polynomial')

class DataSet:
    def __init__(self, wl, t, data, err=None, name=None, freq_unit='nm'):
        """Class for containing a 2D spectra.
        
        Parameters
        ----------
        wl : array of shape(n)
            Array of the spectral dimension
        t : array of shape(m)
            Array with the delay times.
        data : array of shape(n, m)
            Array with the data for each point.
        err : array of shape(n, m) (optional)
            Contains the std err  of the data.
        name : str
            Identifier for data set. (optional)
        freq_unit : {'nm', 'cm'} 
            Unit of the wavelength array, default is 'nm'.
        """

        assert((wl.shape[0], t.shape[0]) == data.shape)

        if freq_unit == 'nm':
            self.wavelengths = wl
            self.wavenumbers = 1e7/wl
            self.wl = self.wavelengths
        else:
            self.wavelengths = 1e7/wl
            self.wavenumbers = wl
            self.wl = self.wavenumbers

        self.t = t
        self.data = data
        if err is not None:
            self.err = err
        if name is not None:
            self.name = name

        #Sort wavelenths and data.
        idx = np.argsort(self.wavelengths)
        self.wavelengths = self.wavelengths[idx]
        self.wavenumbers = self.wavenumbers[idx]
        self.data = self.data[:, idx]


    def __iter__(self):
        """For compatbility with dv.tup"""
        return iter(self.wavelengths, self.t, self.data)

    @staticmethod
    def from_txt(cls, fname, freq_unit='cm', time_div=1000., loadtxt_kws=None):
        """Directly create a dataset from a text file.

        Parameters
        ----------
        fname : str 
            Name of the file. This function assumes the data is given
            by (n+1, m+1) tabular. The first row (exculding the value at [0, 0])
            are frequiencies, the first colum (excluding [0, 0]) are the delay times.
        freq_unit : {'nm', 'cm'}
            Unit of the frequencies.
        time_div : float
            Since `skultrafast` prefers to work with picoseconds and most programms
            use femtoseconds, it divides the time-values. By default it divides
            by `1000`. Use `1` not change the time values.  
        loadtxt_kws : dict
            Dict containing keyword arguments to `np.loadtxt`. 
        """
        if loadtxt_kws is None:
            loadtxt_kws = {}
        tmp = np.loadtxt(fname, loadtxt_kws)
        t = tmp[1:, 0] / time_div
        freq = tmp[0, 1:]
        data = tmp[1:, 1:]
        return cls(freq, t, data, freq_unit=freq_unit)

        
    def save_txt(self, fname, freq_unit='wl'):
        """Save the dataset as a text file
        
        Parameters
        ----------
        fname : str
            Filename (can include filepath)

        freq_unit : 'nm' or 'cm' (default 'nm')
            Which frequency unit is used.
        """
        wl = self.wavelengths if freq_unit is 'wl' else self.wavenumbers
        save_txt(fname, wl, self.t, self.data)

    def cut_freqs(self, freq_ranges=None, freq_unit='nm'):
        """Remove channels outside of given frequency ranges.

        Parameters
        ----------
        freq_ranges : list of (float, float)
            List containing the edges (lower, upper) of the
            frequencies to keep.
        freq_unit : {'nm', 'cm'}
            Unit of the given edges.
        
        Returns
        -------
        : DataSet
            DataSet containing only the listed regions.
        """
        idx = np.zeros_like(self.wavelengths, dtype=np.bool)
        arr =  self.wavelengths if freq_unit is 'nm' else self.wavenumbers
        for (lower, upper) in freq_ranges:           
                idx ^= np.logical_and(arr > lower, arr < upper)
        if self.err is not None:
            err = self.err[:, idx]
        else:
            err = None
        return DataSet(arr[idx], self.t, self.data[:, idx], err, freq_unit)

    def mask_freqs(self, freq_ranges=None, freq_unit='nm'):
        """Mask channels outside of given frequency ranges.

        Parameters
        ----------
        freq_ranges : list of (float, float)
            List containing the edges (lower, upper) of the
            frequencies to keep.
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
        if self.err is not None:
            self.err.mask[:, idx] = True
        self.data.mask[:, idx] = True
        self.wavelengths = np.ma.MaskedArray(self.wavelengths, idx)
        self.wavenumbers = np.ma.MaskedArray(self.wavenumbers, idx)

    def cut_times(self, time_ranges):
        """Remove spectra outside of given time-ranges.

        Parameters
        ----------
        time_ranges : list of (float, float)
            List containing the edges of the time-regions to keep.
        
        Returns
        -------
        : DataSet 
            DataSet containing only the requested regions.
        """
        idx = np.zeros_like(self.t, dtype=np.bool)
        arr = self.t
        for (lower, upper) in time_ranges:           
                idx ^= np.logical_and(arr > lower, arr < upper)
        if self.err is not None:
            err = self.err[idx, :]
        else:
            err = None
        return DataSet(self.wavelengths, self.t[idx], self.data[idx, :], err)

    def mask_times(self, time_ranges):
        """Mask spectra outside of given time-ranges.

        Parameters
        ----------
        time_ranges : list of (float, float)
            List containing the edges of the time-regions to keep.
        
        Returns
        -------
        : None
        """
        idx = np.zeros_like(self.t, dtype=np.bool)
        arr = self.t
        for (lower, upper) in time_ranges:           
                idx ^= np.logical_and(arr > lower, arr < upper)
        if self.err is not None:
            self.err[idx, :].mask = True
        self.t = np.ma.MaskedArray(self.t, idx)
        self.data.mask[:, idx] = True


    def subtract_background(self, n : int=10):
        """Subtracts the first n-spectra from the dataset"""
        self.data -= np.mean(self.data[:n, :], 0, keepdims=1)

    def bin_freqs(self, n : int, freq_unit='nm'):
        """Bins down the dataset by averaging over spectral channel.

        Parameters
        ----------
        n : int
            The number of bins. The edges are calculated by
            np.linspace(freq.min(), freq.max(), n+1).
        freq_unit : {'nm', 'cm'}
            Whether to calculate the bin-borders in
            frequency- of wavelength-space.
        """
        # We use the negative of the wavenumbers to make the array sorted
        arr =  self.wavelengths if freq_unit is 'nm' else -self.wavenumbers
        # Slightly offset edges to include themselves.
        edges = np.linspace(arr.min()-0.002, arr.max()+0.002, n+1)
        idx = np.searchsorted(arr, edges)
        binned = np.empty((self.data.shape[0], n))
        binned_wl = np.empty(n)
        binned_wl = np.empty(n)
        for i in range(n):
            binned[:,i] = np.average(self.data[:,idx[i]:idx[i+1]],1)
            binned_wl[i] = np.mean(arr[idx[i]:idx[i+1]])
        if freq_unit is 'cm':
            binned_wl = - binned_wl
        return DataSet(arr, self.t, binned,freq_unit)

    def estimate_dispersion(self, heuristic='abs', heuristic_args=(1,), deg=2):
        """Estimates the dispersion from a dataset by first
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

        Returns
        -------
        : EstDispResult
            Tuple containing a dispersion correction version of the dataset,
            and array with the estimated time-zeros, and the polynomial function.                     
        """

        if heuristic == 'abs':
            idx = zero_finding.use_first_abs(self.data, heuristic_args[0])

        vals, coefs = zero_finding.robust_fit_tz(self.wl, self.t[idx], deg)
        func = np.polynomial.poly1d(coefs)
        new_data = zero_finding.interpol(self, tn)
        EstDispResult(

    def fit_das(self, x0, fit_t0=False, fix_last_decay=True):
        """Fit expontials to the dataset.
        
        Parameters
        ----------
        x0 : list of floats or array
            Starting values of the fit. The first value is the estimate
            of the system response time omega. If `fit_t0` is true, the second
            float is the guess of the time-zero. All other floats are intepreted
            as the guessing values for expontial decays.        
        fit_t0 : bool (optional)
            If the time-zero should be determined by the fit too. (`True` by default)
        fix_last_decay : bool (optional)
            Fixes the value of the last tau of the inital guess. Is used to add a constant
            contribution by setting the last tau to a large value and fix it. 
        """
        pass


  
    def lft_density_map(self, taus, alpha=1e-4, ): 
        """Calculates the LDM from a dataset by regularized regression.

        Parameters
        """ 
        pass


class DataSetPlotter:
    def __init__(self, dataset):
        """Class for plotting a dataset via matplotlib"""
        self.dataset = dataset


class DataSetInteractiveViewer:
    def __init__(self, dataset, fig_kws={}):
        """Class showing a interactive matplotlib window for exploring
        a dataset.
        """
        import matplotlib.pyplot as plt
        self.dataset = dataset
        self.figure, axs = plt.subplots(3, 1, **fig_kws)
        self.ax_img, self.ax_trans, self.ax_spec = axs
        self.ax_img.pcolormesh(dataset.wl, dataset.t, dataset.data)
        self.trans_line = self.ax_trans.plot()
        self.spec_line = self.ax_spec.plot()

    def init_event(self):
        'Connect mpl events'
        connect = self.figure.canvas.mpl_connect
        connect('motion_notify_event', self.update_lines) 

    def update_lines(self, event):
        """If the mouse cursor is over the 2D image, update
        the dynamic transient and spectrum"""
        
        


    
        
        




           




