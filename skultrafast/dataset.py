from collections import namedtuple
import attr
import lmfit
from lmfit.minimizer import MinimizerResult
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from scipy.interpolate import interp1d, UnivariateSpline
from matplotlib.cbook import deprecated
from matplotlib.lines import Line2D

import skultrafast.dv as dv
from skultrafast.utils import sigma_clip, linreg_std_errors
import skultrafast.plot_helpers as ph
from skultrafast import zero_finding, fitter, lifetimemap
from skultrafast.data_io import save_txt
from skultrafast.kinetic_model import Model
from skultrafast import filter

from typing import Callable, List, Optional, Type, Union, Iterable, Dict
ndarray: Type[np.ndarray] = np.ndarray

EstDispResult = namedtuple("EstDispResult", "correct_ds tn polynomial")
EstDispResult.__doc__ = """
Tuple containing the results from an dispersion estimation.

Attributes
----------
correct_ds : TimeResSpec
    A dataset were we used linear interpolation to remove the dispersion.
tn : array
    Array containing the results of the applied heuristic.
polynomial : function
    Function which maps wavenumbers to time-zeros.
"""

#FitExpResult = namedtuple("FitExpResult", "lmfit_mini lmfit_res fitter")


@attr.s(auto_attribs=True)
class FitExpResult:
    lmfit_mini: lmfit.Minimizer
    lmfit_res: MinimizerResult
    fitter: fitter.Fitter
    pol_resolved: bool = False
    std_errs: Optional[np.ndarray] = None
    var: Optional[np.ndarray] = None
    r2: Optional[np.ndarray] = None

    def calculate_stats(self):
        f = self.fitter
        std_errs, vars, r2s = linreg_std_errors(f.x_vec, f.data)
        self.std_errs = std_errs
        self.vars = vars
        self.r2s = r2s

    def make_sas(self,
                 model: Model,
                 QYs: Dict[str, float] = {},
                 y0: Optional[np.ndarray] = None):
        """
        Generate the species associated spectra from a given model using
        the current das.

        Parameters
        ----------
        model : Model
            Model describing the kinetics. The number of transition rates should
            be identical to the number of DAS-rates. Currtenly, the function
            assumes that the transitions are added in a sorted way, e.g. fastest
            rates first.
        QYs : dict
            Values for the yields.
        y0 : ndarray
            Starting concentrations. If none, y0 = [1, 0, 0, ...].
        Returns
        -------
        ndarry            
        """
        f = self.fitter
        taus = f.last_para[-f.num_exponentials:]
        kvals = 1 / taus
        if y0 is None:
            y0 = np.zeros(len(taus))
            y0[0] = 1
        comps = list(map(str, model.get_compartments()))
        func = model.build_mat_func()
        K = func(*kvals, **QYs)
        vals, vecs = np.linalg.eig(K)
        if np.any(np.subtract.outer(vals, vals) > 1e10):
            raise ValueError("Multivalued eigenvalue")
        vecs = vecs[:, ::-1]
        A = (vecs @ np.diag(np.linalg.solve(vecs, y0))).T
        sas = np.linalg.solve(A, f.c[:, :f.num_exponentials].T)
        ct = f.x_vec[:, :f.num_exponentials] @ A
        return sas, ct


@attr.s(eq=False)
class LDMResult:
    skmodel: object = attr.ib()
    coefs: np.ndarray = attr.ib()
    fit: np.ndarray = attr.ib()
    alpha: np.ndarray = attr.ib()


class TimeResSpec:
    def __init__(
        self,
        wl,
        t,
        data,
        err=None,
        name=None,
        freq_unit="nm",
        disp_freq_unit=None,
        auto_plot=True,
    ):
        """
        Class for working with time-resolved spectra. If offers methods for
        analyzing and pre-processing the data. To visualize the data,
        each `TimeResSpec` object has an instance of an `DataSetPlotter` object
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
        name : str (optional)
            Identifier for data set.
        freq_unit : 'nm' or 'cm' (optional)
            Unit of the wavelength array, default is 'nm'.
        disp_freq_unit : 'nm','cm' or None (optional)
            Unit which is used by default for plotting, masking and cutting
            the dataset. If `None`, it defaults to `freq_unit`.

        Attributes
        ----------
        wavelengths, wavenumbers, t, data : ndarray
            Arrays with the data itself.
        plot : TimeResSpecPlotter
            Helper class which can plot the dataset using `matplotlib`.
        t_idx : function
            Helper function to find the nearest index in t for a given time.
        wl_idx : function
            Helper function to search for the nearest wavelength index for a
            given wavelength.
        wn_idx : function
            Helper function to search for the nearest wavelength index for a
            given wavelength.

        auto_plot : bool
            When True, some function will display their result automatically.            
        """

        assert (
            t.shape[0], wl.shape[0]
        ) == data.shape, f"Data shapes do not match: {t.shape}, {wl.shape} != {data.shape}"
        t = t.copy()
        wl = wl.copy()
        data = data.copy()

        if freq_unit == "nm":
            self._wavelengths = wl
            self._wavenumbers = 1e7 / wl

        else:
            self._wavelengths = 1e7 / wl
            self._wavenumbers = wl

        self.wn = self.wavenumbers
        self.wl = self.wavelengths

        self.t = t
        self.data = data
        self.err = err

        if name is not None:
            self.name = name

        # Sort wavelenths and data.
        idx = np.argsort(self._wavelengths)
        self._wavelengths = self._wavelengths[idx]
        self._wavenumbers = self._wavenumbers[idx]
        self.data = self.data[:, idx]
        if err is not None:
            self.err = self.err[:, idx]
        self.auto_plot = auto_plot
        self.plot = TimeResSpecPlotter(self)
        self.t_idx = lambda x: dv.fi(self.t, x)
        self.wl_idx = lambda x: dv.fi(self.wavelengths, x)
        self.wn_idx = lambda x: dv.fi(self.wavenumbers, x)

        if disp_freq_unit is None:
            self.disp_freq_unit = freq_unit
        else:
            self.disp_freq_unit = disp_freq_unit
        self.plot.freq_unit = self.disp_freq_unit

    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, wavelengths):
        self._wavelengths = wavelengths
        self._wavenumbers = 1e7 / wavelengths

    @property
    def wavenumbers(self):
        return self._wavenumbers

    @wavenumbers.setter
    def wavenumbers(self, wavenumbers):
        self._wavelengths = 1e7 / wavenumbers
        self._wavenumbers = wavenumbers

    def __iter__(self):
        """For compatibility with dv.tup"""
        return iter((self.wavelengths, self.t, self.data))

    def wl_d(self, wl: float):
        """
        Returns the nearest transient for given wavelength.
        """
        idx = self.wl_idx(wl)
        return self.data[:, idx]

    def wn_i(self, wn1, wn2, method='trapz'):
        """
        Integrates the signal from wn1 to wn2

        Parameters
        ----------
        wn1, float
            Wavenumber of the first edge
        wn2, float
            Wavenumber of the second edge
        method, ('trapz', 'spline')
            Method used to integrate.
        """
        wn_min, wn_max = sorted([wn1, wn2])
        idx_min, idx_max = self.wn_idx(wn_min), self.wn_idx(wn_max)
        if (wn_max > self.wavenumbers.max()):
            idx_max = None
        if (wn_min < self.wavenumbers.min()):
            idx_min = None
        sl = slice(idx_min, idx_max)
        x = self.wavenumbers[sl]
        y = self.data[:, sl]
        if method == 'trapz':
            return np.trapz(x=x, y=y, axis=1)
        elif method == 'spline':
            sp = UnivariateSpline(x, y)
            return sp.antiderivative(1)(x)

    def wn_d(self, wn: float):
        """
        Returns the nearest transient for given wavenumber.
        """
        idx = self.wn_idx(wn)
        return self.data[:, idx]

    def t_d(self, t):
        """
        Returns the nearest spectrum for given delaytime.
        """
        idx = self.t_idx(t)
        return self.data[idx, :]

    def copy(self) -> "TimeResSpec":
        """Returns a copy of the TimeResSpec."""
        return TimeResSpec(
            self.wavelengths,
            self.t,
            self.data,
            disp_freq_unit=self.disp_freq_unit,
            err=self.err,
            auto_plot=self.auto_plot,
        )

    @classmethod
    def from_txt(cls,
                 fname,
                 freq_unit="nm",
                 time_div=1.0,
                 transpose=False,
                 disp_freq_unit=None,
                 loadtxt_kws=None):
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
        transpose : bool
            Transposes the loaded array.
        disp_freq_unit : Optional[str]
            See  class documentation.
        loadtxt_kws : dict
            Dict containing keyword arguments to `np.loadtxt`.
        """
        if loadtxt_kws is None:
            loadtxt_kws = {}
        tmp = np.loadtxt(fname, **loadtxt_kws)
        if transpose:
            tmp = tmp.T
        t = tmp[1:, 0] / time_div
        freq = tmp[0, 1:]
        data = tmp[1:, 1:]
        return cls(freq, t, data, freq_unit=freq_unit, disp_freq_unit=disp_freq_unit)

    def save_txt(self, fname, freq_unit="wl"):
        """
        Saves the dataset as a text file.

        Parameters
        ----------
        fname : str
            Filename (can include path)

        freq_unit : 'nm' or 'cm' (default 'nm')
            Which frequency unit is used.
        """
        wl = self.wavelengths if freq_unit == "wl" else self.wavenumbers
        save_txt(fname, wl, self.t, self.data)
        if self.err is not None:
            save_txt(str(fname) + '.stderr', wl, self.t, self.err)

    @deprecated("cut_freqs is deprecated, use cut_freq instead")
    def cut_freqs(self,
                  freq_ranges=None,
                  invert_sel=False,
                  freq_unit=None) -> "TimeResSpec":
        """
        Removes channels inside (or outside ) of given frequency ranges.

        Parameters
        ----------
        freq_ranges : list of (float, float)
            List containing the edges (lower, upper) of the
            frequencies to keep.
        invert_sel : bool
            Invert the final selection.
        freq_unit : 'nm', 'cm' or None
            Unit of the given edges.

        Returns
        -------
        : TimeResSpec
            TimeResSpec containing only the listed regions.
        """
        idx = np.zeros_like(self.wavelengths, dtype=np.bool)
        if freq_unit is None:
            freq_unit = self.disp_freq_unit
        arr = self.wavelengths if freq_unit == "nm" else self.wavenumbers
        for (lower, upper) in freq_ranges:
            idx ^= np.logical_and(arr > lower, arr < upper)
        if not invert_sel:
            idx = ~idx
        if self.err is not None:
            err = self.err[:, idx]
        else:
            err = None
        return TimeResSpec(
            self.wavelengths[idx],
            self.t,
            self.data[:, idx],
            err,
            "nm",
            disp_freq_unit=self.disp_freq_unit,
        )

    def cut_freq(self,
                 lower=-np.inf,
                 upper=np.inf,
                 invert_sel=False,
                 freq_unit=None) -> "TimeResSpec":
        """
        Removes channels inside (or outside ) of given frequency ranges.

        Parameters
        ----------
        lower : float
            Lower bound of the region
        upper : float
            Upper bound of the region
        invert_sel : bool
            Invert the final selection.
        freq_unit : 'nm', 'cm' or None
            Unit of the given edges.

        Returns
        -------
        : TimeResSpec
            TimeResSpec containing only the listed regions.
        """
        idx = np.zeros_like(self.wavelengths, dtype=np.bool)
        if freq_unit is None:
            freq_unit = self.disp_freq_unit
        arr = self.wavelengths if freq_unit == "nm" else self.wavenumbers

        idx ^= np.logical_and(arr > lower, arr < upper)
        if not invert_sel:
            idx = ~idx
        if self.err is not None:
            err = self.err[:, idx]
        else:
            err = None
        return TimeResSpec(
            self.wavelengths[idx],
            self.t,
            self.data[:, idx],
            err,
            "nm",
            disp_freq_unit=self.disp_freq_unit,
        )

    def mask_freq_idx(self, idx):
        """Masks given freq idx array

        Parameters
        ----------
        idx : array
            Boolean array, same shape as the freqs. Where it is
            `True`, the freqs will be masked.
        """
        if self.err is not None:
            self.err = np.ma.MaskedArray(self.err)
            self.err[:, idx] = np.ma.masked
        self.data = np.ma.MaskedArray(self.data)
        self.data[:, idx] = np.ma.masked

    def mask_freqs(self, freq_ranges=None, invert_sel=False, freq_unit=None):
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
        freq_unit : 'nm', 'cm' or None
            Unit of the given edges.

        Returns
        -------
        : None
        """
        idx = np.zeros_like(self.wavelengths, dtype=np.bool)
        if freq_unit is None:
            freq_unit = self.disp_freq_unit
        arr = self.wavelengths if freq_unit == "nm" else self.wavenumbers

        for (lower, upper) in freq_ranges:
            idx ^= np.logical_and(arr > lower, arr < upper)
        if invert_sel:
            idx = ~idx
        if self.err is not None:
            self.err = np.ma.MaskedArray(self.err)
            self.err[:, idx] = np.ma.masked
        self.data = np.ma.MaskedArray(self.data)
        self.data[:, idx] = np.ma.masked

    @deprecated("Use cut_time instead.")
    def cut_times(self, time_ranges, invert_sel=False) -> "TimeResSpec":
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
        : TimeResSpec
            TimeResSpec containing only the requested regions.
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

        return TimeResSpec(
            self.wavelengths,
            self.t[idx],
            self.data[idx, :],
            err,
            "nm",
            disp_freq_unit=self.disp_freq_unit,
        )

    def cut_time(self, lower=-np.inf, upper=np.inf, invert_sel=False) -> "TimeResSpec":
        """
        Remove spectra inside (or outside) of given time-ranges.

        Parameters
        ----------
        lower : float
            Lower bound of the region
        upper : float
            Upper bound of the region
        invert_sel : bool
            Inverts the final selection.
        Returns
        -------
        : TimeResSpec
            TimeResSpec containing only the requested regions.
        """
        idx = np.zeros_like(self.t, dtype=np.bool)
        arr = self.t

        idx ^= np.logical_and(arr > lower, arr < upper)
        if not invert_sel:
            idx = ~idx
        if self.err is not None:
            err = self.err[idx, :]
        else:
            err = None

        return TimeResSpec(
            self.wavelengths,
            self.t[idx],
            self.data[idx, :],
            err,
            "nm",
            disp_freq_unit=self.disp_freq_unit,
        )

    def scale_and_shift(self,
                        scale: float = 1,
                        t_shift: float = 0,
                        wl_shift: float = 0) -> "TimeResSpec":
        """
        Return a dataset which is scaled and/or has shifted times
        and frequencies.

        scale : float
            Scales the whole dataset by given factor.
        t_shift : float
            Shifts the time-axis of an dataset.
        wl_shift : float
            Shifts the wavelengths axis and updates the wavenumbers too.
        
        Returns
        -------
        TimeResSpec
            A modified new dataset
        """
        cpy = self.copy()
        cpy.data *= scale
        cpy.t += t_shift
        cpy.wavelengths += wl_shift
        cpy.wavenumbers = 1e7 / cpy.wavelengths
        return cpy

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
        self.data -= np.mean(self.data[:n, :], 0, keepdims=True)

    def bin_freqs(self, n: int, freq_unit=None, use_err: bool = True) -> "TimeResSpec":
        """
        Bins down the dataset by averaging over several transients.

        Parameters
        ----------
        n : int
            The number of bins. The edges are calculated by
            np.linspace(freq.min(), freq.max(), n+1).
        freq_unit : 'nm', 'cm' or None
            Whether to calculate the bin-borders in frequency- of wavelength
            space. If `None`, it defaults to `self.disp_freq_unit`.
        use_err : bool
            If true, use error for weighting.
        Returns
        -------
        TimeResSpec
            Binned down `TimeResSpec`
        """
        # We use the negative of the wavenumbers to make the array sorted
        if freq_unit is None:
            freq_unit = self.disp_freq_unit
        arr = self.wavelengths if freq_unit == "nm" else -self.wavenumbers
        # Slightly offset edges to include themselves.
        edges = np.linspace(arr.min() - 0.002, arr.max() + 0.002, n + 1)
        idx = np.searchsorted(arr, edges)
        binned = np.empty((self.data.shape[0], n))
        binned_wl = np.empty(n)
        binned_err = np.empty_like(binned)

        for i in range(n):
            if self.err is None or not use_err:
                weights = None
            else:
                weights = 1 / self.err[:, idx[i]:idx[i + 1]]**2
            vals = self.data[:, idx[i]:idx[i + 1]]
            binned[:, i] = np.average(vals, 1, weights=weights)
            if weights is not None:
                binned_err[:, i] = np.average((vals - binned[:, i, None])**2,
                                              1,
                                              weights=weights)
            binned_wl[i] = np.mean(arr[idx[i]:idx[i + 1]])
        if freq_unit == "cm":
            binned_wl = -binned_wl
        if self.err is None or not use_err:
            weights = None

        return TimeResSpec(
            binned_wl,
            self.t,
            binned,
            err=binned_err,
            freq_unit=freq_unit,
            disp_freq_unit=self.disp_freq_unit,
        )

    def bin_times(self, n, start_index=0) -> "TimeResSpec":
        """
        Bins down the dataset by binning `n` sequential spectra together.

        Parameters
        ----------
        n : int
            How many spectra are binned together.
        start_index : int
            Determines the starting index of the binning

        Returns
        -------
        TimeResSpec
            Binned down `TimeResSpec`
        """

        out = []
        out_t = []
        m = len(self.t)
        for i in range(start_index, m, n):
            end_idx = min(i + n, m)
            out.append(
                sigma_clip(self.data[i:end_idx, :], sigma=2.5, max_iter=1,
                           axis=0).mean(0))
            out_t.append(self.t[i:end_idx].mean())

        new_data = np.array(out)
        new_t = np.array(out_t)
        out_ds = self.copy()
        out_ds.t = new_t
        out_ds.data = new_data
        return out_ds

    def estimate_dispersion(self,
                            heuristic="abs",
                            heuristic_args=(),
                            deg=2,
                            shift_result=0,
                            t_parameter=1.3):
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
        shift_result : float
            The resulting dispersion curve is shifted by this value. Default 0.
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

        func_dict = {
            "abs": zero_finding.use_first_abs,
            "diff": zero_finding.use_diff,
            "gauss_diff": zero_finding.use_gaussian,
            "max": zero_finding.use_max,
        }

        if callable(heuristic):
            idx = heuristic(self.t, self.data, *heuristic_args)
        elif heuristic in func_dict:
            idx = func_dict[heuristic](self.data, *heuristic_args)
        else:
            raise ValueError("`heuristic` must be either a callable or"
                             " one of `max`, `abs`, `diff` or `gauss_diff`.")

        vals, coefs = zero_finding.robust_fit_tz(self.wavenumbers,
                                                 self.t[idx],
                                                 deg,
                                                 t=t_parameter)
        coefs[-1] += shift_result
        func = np.poly1d(coefs)
        result = EstDispResult(
            correct_ds=self.interpolate_disp(func),
            tn=self.t[idx] + shift_result,
            polynomial=func,
        )
        if self.auto_plot:
            self.plot.plot_disp_result(result)

        self.disp_result_ = result
        return result

    def interpolate_disp(self, polyfunc: Union[Callable, Iterable]) -> "TimeResSpec":
        """
        Correct for dispersion by linear interpolation .

        Parameters
        ----------
        polyfunc : Union[Callable, Iterable]
            Function which takes wavenumbers and returns time-zeros.

        Returns
        -------
        TimeResSpec
            New TimeResSpec where the data is interpolated so that all channels
            have the same delay point.
        """
        c = self.copy()
        if callable(polyfunc):
            zeros = polyfunc(self.wavenumbers)
        else:
            zeros = polyfunc
        ntc = zero_finding.interpol(self, zeros)
        tmp_tup = dv.tup(self.wavelengths, self.t, self.data)
        ntc_err = zero_finding.interpol(tmp_tup, zeros)
        c.data = ntc.data
        c.err = ntc_err.data
        return c

    def fit_exp(
        self,
        x0,
        fix_sigma=True,
        fix_t0=True,
        fix_last_decay=True,
        model_coh=False,
        lower_bound=0.1,
        verbose=True,
        use_error=False,
        fixed_names=None,
    ):
        """
        Fit a sum of exponentials to the dataset. This function assumes
        the dataset is already corrected for dispersion.

        Parameters
        ----------
        x0 : list of floats or array
            Starting values of the fit. The first value is the estimate of the
            system response time omega. If `fit_t0` is true, the second float is
            the guess of the time-zero. All other floats are interpreted as the
            guessing values for exponential decays.
        fix_sigma : bool (optional)
            If to fix the IRF duration sigma.
        fix_t0 : bool (optional)
            If to fix the the time-zero.
        fix_last_decay : bool (optional)
            Fixes the value of the last tau of the initial guess. It can be
            used to add a constant by setting the last tau to a large value
            and fix it.
        model_coh : bool (optional)
            If coherent contributions should by modeled. If `True` a gaussian
            with a width equal the system response time and its derivatives are
            added to the linear model.
        lower_bound : float (optional)
            Lower bound for decay-constants.
        verbose : bool
            Prints the results out if True.
        use_error : bool
            If the errors are used in the fit.
        fixed_names : list of str
            Can be used to fix time-constants
        """

        f = fitter.Fitter(self, model_coh=model_coh, model_disp=1)
        if use_error:
            f.weights = 1 / self.err
        f.res(x0)

        if fixed_names is None:
            fixed_names = list()
        if fix_sigma:
            fixed_names.append("w")

        lm_model = f.start_lmfit(
            x0,
            fix_long=fix_last_decay,
            fix_disp=fix_t0,
            lower_bound=lower_bound,
            full_model=False,
            fixed_names=fixed_names,
        )
        ridge_alpha = abs(self.data).max() * 1e-4
        f.lsq_method = "ridge"
        fitter.alpha = ridge_alpha
        result = lm_model.leastsq()
        result_tuple = FitExpResult(lm_model, result, f)
        result_tuple.calculate_stats()
        self.fit_exp_result_ = result_tuple
        if verbose:
            lmfit.fit_report(result)
        return result_tuple

    def lifetime_density_map(self,
                             taus=None,
                             alpha=1e-4,
                             cv=True,
                             maxiter=30000,
                             **kwargs):
        """Calculates the LDM from a dataset by regularized regression.

        Parameters
        ----------
        taus : array or None
            List with potential decays for building the basis. If `None`,
            use automatic determination.
        alpha : float
            The regularization factor.
        cv : bool
            If to apply cross-validation, by default True.
        """
        if taus is None:
            dt = self.t[self.t_idx(0)] - self.t[self.t_idx(0) - 1]
            max_t = self.t.max()
            start = np.floor(np.log10(dt))
            end = np.ceil(np.log10(max_t))
            taus = np.geomspace(start, end, 5 * (end-start))

        result = lifetimemap.start_ltm(self,
                                       taus,
                                       use_cv=cv,
                                       add_const=False,
                                       alpha=alpha,
                                       add_coh=False,
                                       max_iter=30000,
                                       **kwargs)
        result = LDMResult(*result)
        return result

    def concat_datasets(self, other_ds):
        """
        Merge the dataset with another dataset. The other dataset need to
        have the same time axis.

        Parameters
        ----------
        other_ds : TimeResSpec
            The dataset to merge with

        Returns
        -------
        TimeResSpec
            The merged dataset.
        """

        all_wls = np.hstack((self.wavelengths, other_ds.wavelengths))
        all_data = np.hstack((self.data, other_ds.data))
        if not (self.err is None or other_ds.err is None):
            all_err = np.hstack((self.err, other_ds.err))
        else:
            all_err = None

        return TimeResSpec(
            all_wls,
            self.t,
            all_data,
            err=all_err,
            freq_unit="nm",
            disp_freq_unit=self.disp_freq_unit,
        )

    def merge_nearby_channels(self,
                              distance: float = 8,
                              use_err: bool = False) -> "TimeResSpec":
        """Merges sequetential channels together if their distance
        is smaller than given.

        Parameters
        ----------
        distance : float, optional
            The minimal distance allowed between two channels. If smaller,
            they will be merged together, by default 8.
        use_err : bool

        Returns
        -------
        TimeResSpec
            The merged dataset.
        """
        skiplist = []
        nwl = self.wavelengths.copy()
        nspec = self.data.copy()
        nerr = self.err.copy() if self.err is not None else None
        weights = 1 / self.err.copy() if self.err is not None else None

        for i in range(nwl.size - 1):
            if i in skiplist:
                continue
            if abs(nwl[i + 1] - nwl[i]) < distance:
                if self.err is not None:
                    if self.err is not None and use_err:
                        w = weights[:, i:i + 2]**2
                    else:
                        w = None
                    mean = np.average(nspec[:, i:i + 2], 1, weights=w)
                    err = np.sqrt(
                        np.average((nspec[:, i:i + 2] - mean[:, None])**2, 1, weights=w))
                    nspec[:, i] = mean
                    if nerr is not None:
                        nerr[:, i] = err

                nwl[i] = np.mean(nwl[i:i + 2])
                skiplist.append(i + 1)

        nwl = np.delete(nwl, skiplist)
        nspec = np.delete(nspec, skiplist, axis=1)
        if nerr is not None:
            nerr = np.delete(nerr, skiplist, axis=1)
        new_ds = self.copy()
        if nerr is not None and use_err:
            new_ds.err = nerr
        else:
            new_ds.err = None
        new_ds.wavelengths = nwl
        new_ds.wavenumbers = 1e7 / nwl
        new_ds.data = nspec
        return new_ds

    def apply_filter(self, kind, args) -> 'TimeResSpec':
        """Apply a filter to the data. Will always return
        a copy of the data.

        Returns
        -------
        kind: callable or in ('svd', 'uniform', 'gaussian')
            What kind of filter to use. Either a string 
            indicating a inbuild filter or a callable.            
        args: any
            Argument to the filter. Depends on the kind.
        """
        filtered_ds = self.copy()
        if callable(kind):
            tup = kind(filtered_ds.data, *args)
        elif kind == 'svd':
            tup = filter.svd_filter(filtered_ds, args)
        elif kind == 'uniform':
            tup = filter.uniform_filter(filtered_ds, args)
        elif kind == "gaussian":
            tup = filter.gaussian_filter(filtered_ds, args)
        filtered_ds.data = tup.data
        return filtered_ds


class PolTRSpec:
    def __init__(self,
                 para: TimeResSpec,
                 perp: TimeResSpec,
                 iso: Optional[TimeResSpec] = None):
        """
        Class for working with a polazation resolved datasets. Assumes the same
        frequency and time axis for both polarisations.

        Parameters
        ----------
        para : TimeResSpec
            The dataset with parallel pump/probe pol.
        perp : TimeResSpec
            The TimeResSpec with perpendicular pump/probe
        iso : Optional[TimeResSpec]
            Iso dataset, if none it will be calculated from para and perp.

        Attributes
        ----------
        plot : PolDataSetPlotter
            Helper class containing the plotting methods.
        """

        assert para.data.shape == perp.data.shape
        self.para = para
        self.perp = perp
        if iso is None:
            self.iso = para.copy()
            self.iso.data = (2 * perp.data + para.data) / 3
        else:
            self.iso = iso
        self.wavenumbers = para.wavenumbers
        self.wavelengths = para.wavelengths
        self.wn, self.wl = self.wavenumbers, self.wavelengths
        self.t = para.t
        self.disp_freq_unit = para.disp_freq_unit
        self.plot = PolTRSpecPlotter(self, self.disp_freq_unit)

        trs = TimeResSpec
        self._copy = delegator(self, trs.copy)
        self.bin_times = delegator(self, trs.bin_times)
        self.bin_freqs = delegator(self, trs.bin_freqs)
        self.cut_times = delegator(self, trs.cut_times)
        self.cut_time = delegator(self, trs.cut_time)
        self.scale_and_shift = delegator(self, trs.scale_and_shift)
        self.cut_freqs = delegator(self, trs.cut_freqs)
        self.cut_freq = delegator(self, trs.cut_freq)
        self.mask_freqs = delegator(self, trs.mask_freqs)
        self.mask_times = delegator(self, trs.mask_times)
        self.subtract_background = delegator(self, trs.subtract_background)
        self.merge_nearby_channels = delegator(self, trs.merge_nearby_channels)
        self.interpolate_disp = delegator(self, trs.interpolate_disp)
        self.apply_filter = delegator(self, trs.apply_filter)
        self.t_idx = para.t_idx
        self.wn_idx = para.wn_idx
        self.wl_idx = para.wl_idx

    def copy(self) -> 'PolTRSpec':
        new_ds: PolTRSpec = self._copy()
        new_ds.plot.para_ls = self.plot.para_ls
        new_ds.plot.perp_ls = self.plot.perp_ls
        return new_ds

    def wl_d(self, wl):
        idx = self.wl_idx(wl)
        return self.para[:, idx], self.perp[:, idx]

    def wn_d(self, wn):
        idx = self.wn_idx(wn)
        return self.para[:, idx], self.perp[:, idx]

    def t_d(self, t):
        idx = self.t_idx(t)
        return self.para.T[:, idx], self.perp.T[:, idx]

    def fit_exp(
        self,
        x0,
        fix_sigma=True,
        fix_t0=True,
        fix_last_decay=True,
        from_t=None,
        model_coh=False,
        lower_bound=0.1,
        use_error=False,
        fixed_names=None,
    ) -> FitExpResult:
        """
        Fit a sum of exponentials to the dataset. This function assumes
        the two datasets is already corrected for dispersion.

        Parameters
        ----------
        x0 : list of floats or array
            Starting values of the fit.  The first value is the guess of the time-zero.
            The second value is the estimate of the system response time omega. If
            `fit_t0` is true, All other floats are interpreted as the guessing values
             for exponential decays.
        fix_sigma : bool (optional)
            If to fix the IRF duration sigma.
        fix_t0 : bool (optional)
            If to fix the the time-zero.
        fix_last_decay : bool (optional)
            Fixes the value of the last tau of the initial guess. It can be
            used to add a constant by setting the last tau to a large value
            and fix it.
        from_t : float or None
            If not None, data with t<from_t will be ignored for the fit.
        model_coh : bool (optional)
            If coherent contributions should by modeled. If `True` a gaussian
            with a width equal the system response time and its derivatives are
            added to the linear model.
        lower_bound : float (optional)
            Lower bound for decay-constants.
        use_error : bool
            Wether to use the error to weight the residuals
        fixed_names : list of str
            Can be used to fix names.
        """
        pa, pe = self.para, self.perp
        if not from_t is None:
            pa = pa.cut_times([(-np.inf, from_t)])
            pe = pe.cut_times([(-np.inf, from_t)])
        all_data = np.hstack((pa.data, pe.data))
        all_wls = np.hstack((pa.wavelengths, pe.wavelengths))
        all_tup = dv.tup(all_wls, pa.t, all_data)
        f = fitter.Fitter(all_tup, model_coh=model_coh, model_disp=1)
        if use_error:
            all_err = np.hstack((pa.err, pe.err))
            f.weights = 1 / all_err
        f.res(x0)
        if fixed_names is None:
            fixed_names = []

        if fix_sigma:
            fixed_names.append("w")

        lm_model = f.start_lmfit(
            x0,
            fix_long=fix_last_decay,
            fix_disp=fix_t0,
            lower_bound=lower_bound,
            full_model=False,
            fixed_names=fixed_names,
        )
        ridge_alpha = abs(all_data).max() * 1e-4
        f.lsq_method = "ridge"
        fitter.alpha = ridge_alpha
        result = lm_model.leastsq()

        self.fit_exp_result_ = FitExpResult(lm_model, result, f)
        self.fit_exp_result_.calculate_stats()
        return self.fit_exp_result_

    def save_txt(self, fname, freq_unit="wl"):
        """
        Saves the dataset as a text file.

        Parameters
        ----------
        fname : str
            Filename (can include path). This functions adds `_para.txt` and
            '_perp.txt' for the corresponding dataset to the fname.
        freq_unit : 'nm' or 'cm' (default 'nm')
            Which frequency unit is used.
        """
        fname = Path(fname)
        self.para.save_txt(fname.with_suffix('.para.txt'), freq_unit)
        self.perp.save_txt(fname.with_suffix('.perp.txt'), freq_unit)
        self.iso.save_txt(fname.with_suffix('.iso.txt'), freq_unit)


import functools
import typing


def delegator(pol_tr: PolTRSpec,
              method: typing.Callable) -> typing.Callable[..., Optional[PolTRSpec]]:
    """
    Helper function to delegate methods calls from PolTRSpec to
    the methods of TimeResSpec.

    Parameters
    ----------
    pol_tr : PolTRSpec
    method : method of TimeResSpec
        The method to wrap. Uses function annotations to check if the
        method returns a new TimeResSpec.
    """
    name = method.__name__
    hints = typing.get_type_hints(method)
    if "return" in hints:
        do_return = hints["return"] == TimeResSpec
    else:
        do_return = False

    @functools.wraps(method)
    def func(*args, **kwargs) -> Optional[PolTRSpec]:
        para = method(pol_tr.para, *args, **kwargs)
        perp = method(pol_tr.perp, *args, **kwargs)
        iso = method(pol_tr.iso, *args, **kwargs)
        if do_return:
            return PolTRSpec(para, perp, iso=iso)
        else:
            return None

    func.__doc__ = method.__doc__
    func.__name__ = name
    return func


class PlotterMixin:
    @property
    def x(self):
        if self.freq_unit == "cm":
            return self._get_wn()
        else:
            return self._get_wl()

    def lbl_spec(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.set_xlabel(ph.freq_label)
        ax.set_ylabel(ph.sig_label)
        ax.autoscale(1, "x", 1)
        ax.axhline(0, color="k", lw=0.5, zorder=1.9)
        ax.legend(loc='best', ncol=2, title='Delay time')
        ax.minorticks_on()

    def upsample_spec(self, y, kind='cubic', factor=4):

        x = self.x
        assert (y.shape[0] == x.size)

        inter = interp1d(x, y, kind=kind, assume_sorted=False)
        fac = factor + 1
        diff = np.diff(x) / (fac)
        new_points = x[:-1, None] + np.arange(1, fac)[None, :] * diff[:, None]
        xn = np.sort(np.concatenate((x, new_points.ravel())))
        return xn, inter(xn)

    def univariate_spline(self, y):
        if self.dataset.err is not None:
            w = 1 / self.dataset.err

        UnivariateSpline(x=self.x, y=y, w=w)


class TimeResSpecPlotter(PlotterMixin):
    _ds_name = "self.pol_ds.para"

    def __init__(self, dataset: TimeResSpec, disp_freq_unit="nm"):
        """
        Class which can Plot a `TimeResSpec` using matplotlib.

        Parameters
        ----------
        dataset : TimeResSpec
            The TimeResSpec to work with.
        disp_freq_unit : {'nm', 'cm'} (optional)
            The default unit of the plots. To change
            the unit afterwards, set the attribute directly.
        """
        self.dataset = dataset
        self.dataset.trans = self.trans
        self.dataset.spec = self.spec
        self.dataset.map = self.map

        self.freq_unit = disp_freq_unit

    def _get_wl(self):
        return self.dataset.wavelengths

    def _get_wn(self):
        return self.dataset.wavenumbers

    def map(self,
            symlog=True,
            equal_limits=True,
            plot_con=True,
            con_step=None,
            con_filter=None,
            ax=None,
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
        con_filter : None, int or `TimeResSpec`.
            Since contours are strongly affected by noise, it can be prefered to
            filter the dataset before calculating the contours. If `con_filter`
            is a dataset, the data of that set will be used for the contours. If
            it is a tuple of int, the data will be filtered with an
            uniform filter before calculation the contours. If `None`, no data
            prepossessing will be applied.
        ax : plt.Axis or None
            Takes a matplotlib axis. If none, it uses `plt.gca()` to get the
            current axes. The lines are plotted in this axis.

        """
        if ax is None:
            ax = plt.gca()
        is_nm = self.freq_unit == "nm"
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()

        ds = self.dataset
        x = ds.wavelengths if is_nm else ds.wavenumbers
        cmap = kwargs.pop("colormap", "bwr")
        if equal_limits:
            m = np.max(np.abs(ds.data))
            vmin, vmax = -m, m
        else:
            vmin, vmax = ds.data.max(), ds.data.min()
        mesh = ax.pcolormesh(x, ds.t, ds.data, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        if symlog:
            ax.set_yscale("symlog", linthresh=1)
            ph.symticks(ax, axis="y")
            ax.set_ylim(-0.5)
        plt.colorbar(mesh, ax=ax)

        con = None
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

            if isinstance(con_filter, TimeResSpec):
                data = con_filter.data
            elif con_filter is not None:  # must be int or tuple of int
                if isinstance(con_filter, tuple):
                    data = filter.uniform_filter(ds, con_filter).data
                else:
                    data = filter.svd_filter(ds, con_filter).data
            else:
                data = ds.data
            con = ax.contour(
                x,
                ds.t,
                data,
                levels=levels,
                linestyles="solid",
                colors="k",
                linewidths=0.5,
            )
        ph.lbl_map(ax, symlog)
        if not is_nm:
            ax.set_xlim(*ax.get_xlim()[::-1])
        return mesh, con

    def spec(self,
             *args,
             norm=False,
             ax=None,
             n_average=0,
             upsample=1,
             use_weights=False,
             offset=0.,
             **kwargs):
        """
        Plot spectra at given times.

        Parameters
        ----------
        *args : list or ndarray
            List of the times where the spectra are plotted.
        norm : bool
            If true, each spectral will be normalized.
        ax : plt.Axis or None.
            Axis where the spectra are plotted. If none, the current axis will
            be used.
        n_average : int
            For noisy data it may be preferred to average multiple spectra
            together. This function plots the average of `n_average` spectra
            around the specific time-points.
        upsample : int,
            If upsample is >1, it will plot an upsampled version of the spectrum
            using cubic spline interplotation.
        use_weights : bool
            If given a tuple, the function will plot the average of the given range.
            use_weights determines if error weights are in calculating the average.
        offset: float or 'auto'
            If non-zero, each spectrum will be shifted by 'offset' relatively to the last one.
            'auto' is not yet implemented.

        Returns
        -------
        list of `Lines2D`
            List containing the Line2D objects belonging to the spectra.
        """
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        if ax is None:
            ax = plt.gca()
        is_nm = self.freq_unit == "nm"
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()

        cur_offset = 0.
        ds = self.dataset
        x = ds.wavelengths if is_nm else ds.wavenumbers
        li = []
        for i in args:
            if isinstance(i, tuple):
                if ds.err is not None and use_weights:
                    weights = 1 / ds.err[ds.t_idx(i[0]):ds.t_idx(i[1]), :]**2
                else:
                    weights = None
                dat = np.average(ds.data[ds.t_idx(i[0]):ds.t_idx(i[1]), :],
                                 weights=weights,
                                 axis=0)
                label = '%.1f ps to %.1f ps' % (i[0], i[1])
            else:
                idx = dv.fi(ds.t, i)
                if n_average > 0:
                    dat = filter.uniform_filter(ds, (2*n_average + 1, 1)).data[idx, :]
                elif n_average == 0:
                    dat = ds.data[idx, :]
                else:
                    raise ValueError("n_average must be an Integer >= 0.")
                label = ph.time_formatter(ds.t[idx], ph.time_unit)
            if upsample > 1:
                x, dat = self.upsample_spec(dat, factor=upsample)
            if norm:
                dat = dat / abs(dat).max()
            markevery = None if upsample == 1 else upsample + 1

            li += ax.plot(x, dat + cur_offset, markevery=markevery, label=label, **kwargs)
            cur_offset += offset
        self.lbl_spec(ax)
        if not is_nm:
            ax.set_xlim(x.max(), x.min())
        return li

    def trans_integrals(self,
                        *args,
                        symlog: bool = True,
                        norm=False,
                        ax=None,
                        **kwargs) -> typing.List[plt.Line2D]:
        """
        Plot the transients of integrated region. The integration will use np.trapz in
        wavenumber-space.

        Parameters
        ----------
        args : tuples of floats
            Tuple of wavenumbers determining the region to be integrated.
        symlog : bool
            If to use a symlog scale for the delay-time.
        norm : bool or float
            If `true`, normalize to transients. If it is a float, the transients are
            normalzied to value at the delaytime norm.
        ax : plt.Axes or None
            Takes a matplotlib axes. If none, it uses `plt.gca()` to get the
            current axes. The lines are plotted in this ax

        kwargs : Further arguments passed to plt.plot

        Returns
        -------
        list of Line2D
            List containing the plotted lines.
        """
        if ax is None:
            ax = plt.gca()
        ph.ir_mode()
        ds = self.dataset
        lines = []
        for (a, b) in args:
            a, b = sorted([a, b])
            idx = (a < ds.wavenumbers) & (ds.wavenumbers < b)
            dat = np.trapz(-ds.data[:, idx], ds.wavenumbers[idx], axis=1)

            if norm is True:
                dat = np.sign(dat[np.argmax(abs(dat))]) * dat / abs(dat).max()
            elif norm is False:
                pass
            else:
                dat = dat / dat[ds.t_idx(norm)]
            lines.extend(ax.plot(ds.t, dat, label=f"{a: .0f} cm-1 to {b: .0f}", **kwargs))

        if symlog:
            ax.set_xscale("symlog", linthresh=1.0)
        ph.lbl_trans(ax=ax, use_symlog=symlog)
        ax.legend(loc="best", ncol=2)
        ax.set_xlim(right=ds.t.max())
        ax.yaxis.set_tick_params(which="minor", left=True)
        return lines

    def trans(self,
              *args,
              symlog=True,
              norm=False,
              ax=None,
              freq_unit="auto",
              linscale=1,
              **kwargs):
        """
        Plot the nearest transients for given frequencies.

        Parameters
        ----------
        *args : list or ndarray
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
        freq_unit : 'auto', 'cm' or 'nm'
            How to interpret the given frequencies. If 'auto' it defaults to
            the plotters freq_unit.
        linscale : float
            If symlog is True, determines the ratio of linear to log-space.

        All other kwargs are forwarded to the plot function.

        Returns
        -------
         list of Line2D
            List containing the plotted lines.
        """

        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]

        if ax is None:
            ax = plt.gca()

        tmp = self.freq_unit if freq_unit == "auto" else freq_unit
        is_nm = tmp == "nm"
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        ds = self.dataset
        x = ds.wavelengths if is_nm else ds.wavenumbers

        t, d = ds.t, ds.data
        l, plotted_vals = [], []
        for i in args:
            idx = dv.fi(x, i)

            dat = d[:, idx]
            if norm is True:
                dat = np.sign(dat[np.argmax(abs(dat))]) * dat / abs(dat).max()
            elif norm is False:
                pass
            else:
                dat = dat / dat[dv.fi(t, norm)]
            plotted_vals.append(dat)
            l.extend(ax.plot(t, dat, label="%.0f %s" % (x[idx], ph.freq_unit), **kwargs))

        if symlog:
            ax.set_xscale("symlog", linthresh=1.0, linscale=linscale)
        ph.lbl_trans(ax=ax, use_symlog=symlog)
        ax.legend(loc="best", ncol=max(1, len(l) // 3))
        ax.set_xlim(right=t.max())
        ax.yaxis.set_tick_params(which="minor", left=True)
        return l

    def overview(self):
        """
        Plots an overview figure.
        """
        is_nm = self.freq_unit == "nm"
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        ds = self.dataset
        x = ds.wavelengths if is_nm else ds.wavenumbers
        fig, axs = plt.subplots(3,
                                1,
                                figsize=(5, 12),
                                gridspec_kw=dict(height_ratios=(2, 1, 1)))
        self.map(ax=axs[0])

        times = np.hstack((0, np.geomspace(0.1, ds.t.max(), 6)))
        sp = self.spec(times, ax=axs[1])
        freqs = np.unique(np.linspace(x.min(), x.max(), 6))
        tr = self.trans(freqs, ax=axs[2])
        OverviewPlot = namedtuple("OverviewPlot", "fig axs trans spec")
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
        is_nm = self.freq_unit == "nm"
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        ds = self.dataset
        x = ds.wavelengths if is_nm else ds.wavenumbers
        fig, axs = plt.subplots(3, 1, figsize=(4, 5))
        u, s, v = np.linalg.svd(ds.data)
        axs[0].stem(s, use_line_collection=True)
        axs[0].set_xlim(0, 11)
        try:
            len(n)
            comps = n
        except TypeError:
            comps = range(n)

        for i in comps:
            axs[1].plot(ds.t, u.T[i], label="%d" % i)
            axs[2].plot(x, v[i])
        ph.lbl_trans(axs[1], use_symlog=True)
        self.lbl_spec(axs[2])

    def das(self, first_comp=0, ax=None, **kwargs):
        """
        Plot a DAS, if available.

        Parameters
        ----------
        fist_comp : int
            Index of the first shown component, useful if 
            fast components model coherent artefact and should
            not be shown
        ax : plt.Axes or None
            Axes to plot.
        kwargs : dict
            Keyword args given to the plot function

        Returns
        -------
        Tuple of (List of Lines2D)
        """
        ds = self.dataset
        if not hasattr(ds, "fit_exp_result_"):
            raise ValueError("The PolTRSpec must have successfully fit the " "data first")
        if ax is None:
            ax = plt.gca()
        is_nm = self.freq_unit == "nm"
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        f = ds.fit_exp_result_.fitter
        num_exp = f.num_exponentials
        leg_text = [ph.nsf(i) + " " + ph.time_unit for i in f.last_para[-num_exp:]]
        if max(f.last_para) > 5 * f.t.max():
            leg_text[-1] = "const."

        l1 = ax.plot(self.x, f.c[:, first_comp:num_exp], **kwargs)
        for i, l in enumerate(l1):
            l.set_label(leg_text[i + first_comp])
        ax.legend(title="Decay\nConstants")
        ph.lbl_spec(ax)
        return l1

    def edas(self, ax=None, legend=True, **kwargs):
        """
        Plot a EDAS, if expontial fit is available.

        Parameters
        ----------
        ax : plt.Axes or None
            Axes to plot.
        kwargs : dict
            Keyword args given to the plot function

        Returns
        -------
        Tuple of (List of Lines2D)
        """
        ds = self.dataset
        if not hasattr(ds, "fit_exp_result_"):
            raise ValueError("The PolTRSpec must have successfully fit the " "data first")
        if ax is None:
            ax = plt.gca()
        is_nm = self.freq_unit == "nm"
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        f = ds.fit_exp_result_.fitter
        num_exp = f.num_exponentials
        taus = f.last_para[-num_exp:]
        das = f.c[:, :num_exp]
        print(np.diff(taus))
        if np.any(np.diff(taus)<0):
            raise ValueError("EADS expected sorted time-constants")

        leg_text = [ph.nsf(i) + " " + ph.time_unit for i in f.last_para[-num_exp:]]
        if max(f.last_para) > 5 * f.t.max():
            leg_text[-1] = "const."
        edas = np.cumsum(das[:, ::-1], axis=1)
        l1 = ax.plot(self.x, edas[:, ::-1], **kwargs)
        
        for i, l in enumerate(l1):
            l.set_label(leg_text[i])
        if legend:
            ax.legend(title="Species")
        ph.lbl_spec(ax)
        return l1

        

    def interactive(self):
        """
        Generates a jupyter widgets UI for exploring a spectra.
        """
        import ipywidgets as wid
        from IPython.display import display

        is_nm = self.freq_unit == "nm"
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        ds = self.dataset
        x = ds.wavelengths if is_nm else ds.wavenumbers
        # fig, ax = plt.subplots()
        wl_slider = wid.FloatSlider(
            None,
            min=x.min(),
            max=x.max(),
            step=1,
            description="Freq (%s)" % self.freq_unit,
        )

        def func(x):

            # ax.cla()
            self.trans([x])
            # plt.show()

        ui = wid.interactive(func, x=wl_slider, continuous_update=False)
        display(ui)
        return ui

    def plot_disp_result(self, result: EstDispResult):
        """Visualize the result of a dispersion correction, creates a figure"""

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col", figsize=(3, 4))
        ds = self.dataset
        tmp_unit = self.freq_unit, result.correct_ds.plot.freq_unit
        self.freq_unit = "cm"
        result.correct_ds.plot.freq_unit = "cm"
        self.map(symlog=False, plot_con=False, ax=ax1)
        ylim = max(ds.t.min(), -2), min(2, ds.t.max())
        ax1.set_ylim(*ylim)
        ax1.plot(ds.wavenumbers, result.tn)
        ax1.plot(ds.wavenumbers, result.polynomial(ds.wavenumbers))

        result.correct_ds.map(symlog=True, con_filter=3, con_step=None)
        self.freq_unit = tmp_unit[0]
        result.correct_ds.plot.freq_unit = tmp_unit[1]


class PolTRSpecPlotter(PlotterMixin):

    perp_ls = dict(marker='s', markersize=3, linewidth=1, markerfacecolor='w')
    para_ls = dict(marker='o', markersize=3, linewidth=1)

    def __init__(self, pol_dataset: PolTRSpec, disp_freq_unit=None):
        """
        Plotting commands for a PolTRSpec

        Parameters
        ----------
        pol_dataset : PolTRSpec
            The Data
        disp_freq_unit : {'nm', 'cm'} (optional)
            The default unit of the plots. To change
            the unit afterwards, set the attribute directly.
        """
        self.pol_ds = pol_dataset
        if disp_freq_unit is not None:
            self.freq_unit = disp_freq_unit

        self.perp_ls = PolTRSpecPlotter.perp_ls.copy()
        self.para_ls = PolTRSpecPlotter.para_ls.copy()

    def _get_wl(self):
        return self.pol_ds.para.wavelengths

    def _get_wn(self):
        return self.pol_ds.para.wavenumbers

    def spec(self, *times, norm=False, ax=None, n_average=0, **kwargs):
        """
        Plot spectra at given times.

        Parameters
        ----------
        *times : list or ndarray
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
        upsample : int
            If >1, upsample the spectrum using cubic interpolation.

        Returns
        -------
        tuple of (List of `Lines2D`)
            List containing the Line2D objects belonging to the spectra.
        """
        if ax is None:
            ax = plt.gca()
        pa, pe = self.pol_ds.para, self.pol_ds.perp
        l1 = pa.plot.spec(*times,
                          norm=norm,
                          ax=ax,
                          n_average=n_average,
                          **self.para_ls,
                          **kwargs)
        l2 = pe.plot.spec(*times,
                          norm=norm,
                          ax=ax,
                          n_average=n_average,
                          **self.perp_ls,
                          **kwargs)
        dv.equal_color(l1, l2)

        colored_lines = [
            Line2D([0], [0], color=l.get_color(), label=l.get_label()) for l in l1
        ]
        pol_lines = [
            Line2D([0], [0], color='0.3', label=r'$\parallel$-pol.', **self.para_ls),
            Line2D([0], [0], color='0.3', label=r'$\perp$-pol.', **self.perp_ls)
        ]
        all_lines = colored_lines + pol_lines
        self.lbl_spec(ax)
        ax.legend(all_lines, [l.get_label() for l in all_lines])
        return l1, l2

    def trans(self, *args, symlog=True, norm=False, ax=None, **kwargs):
        """
        Plot the nearest transients for given frequencies.

        Parameters
        ----------
        wls : list or ndarray
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
            Tuple of lists containing the plotted lines.
        """
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        if ax is None:
            ax = plt.gca()
        pa, pe = self.pol_ds.para, self.pol_ds.perp

        # Avoid duplicated keywords
        duplicated_para = {}
        duplicated_perp = {}

        for k in kwargs:
            if k in self.para_ls:
                self.para_ls.pop(k)
                duplicated_para[k] = kwargs[k]

            if k in self.perp_ls:
                self.perp_ls.pop(k)
                duplicated_perp[k] = kwargs[k]

        l1 = pa.plot.trans(*args,
                           symlog=symlog,
                           norm=norm,
                           ax=ax,
                           **kwargs,
                           **self.para_ls)
        l2 = pe.plot.trans(*args,
                           symlog=symlog,
                           norm=norm,
                           ax=ax,
                           **kwargs,
                           **self.perp_ls)

        self.para_ls.update(**duplicated_para)
        self.para_ls.update(**duplicated_perp)
        dv.equal_color(l1, l2)

        colored_lines = [
            Line2D([0], [0], color=l.get_color(), label=l.get_label()) for l in l1
        ]
        pol_lines = [
            Line2D([0], [0], color='0.3', label=r'$\parallel$-pol.', **self.para_ls),
            Line2D([0], [0], color='0.3', label=r'$\perp$-pol.', **self.perp_ls)
        ]
        all_lines = colored_lines + pol_lines

        ax.legend(all_lines, [l.get_label() for l in all_lines])

        return l1, l2

    def das(self, ax=None, plot_first_das=True, **kwargs):
        """
        Plot a DAS, if available.

        Parameters
        ----------
        ax : plt.Axes or None
            Axes to plot.
        plot_first_das : bool
            If true, the first DAS is omitted. This is useful, when the first
            component is very fast and only modeles coherent contributions.
        kwargs : dict
            Keyword args given to the plot function

        Returns
        -------
        Tuple of (List of Lines2D)
        """
        ds = self.pol_ds
        if not hasattr(self.pol_ds, "fit_exp_result_"):
            raise ValueError("The PolTRSpec must have successfully fit the " "data")
        if ax is None:
            ax = plt.gca()
        is_nm = self.freq_unit == "nm"

        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        f = ds.fit_exp_result_.fitter
        num_exp = f.num_exponentials
        leg_text = [ph.nsf(i) + " " + ph.time_unit for i in f.last_para[-num_exp:]]
        if max(f.last_para) > 5 * f.t.max():
            leg_text[-1] = "const."
        n = ds.para.wavelengths.size
        x = ds.para.wavelengths if is_nm else ds.para.wavenumbers
        start = 0 if plot_first_das else 1
        palines = []
        pelines = []
        for c, i in enumerate(range(start, num_exp)):
            l1 = ax.plot(x,
                         f.c[:n, i],
                         **kwargs,
                         **self.para_ls,
                         label=leg_text[i],
                         color='C%d' % c)
            l2 = ax.plot(x, f.c[n:, i], **kwargs, **self.perp_ls)
            dv.equal_color(l1, l2)
            palines += l1
            pelines += l2
        ph.lbl_spec(ax=ax)
        ncol = max(num_exp // 3, 1)
        ax.legend(title="Decay\nConstants", ncol=ncol)
        return palines, pelines

    def edas(self, ax=None, *, add_legend=True, **kwargs):
        """
        Plots a SAS (also called EDAS), if available.

        Parameters
        ----------
        ax : plt.Axes or None
            Axes to plot.
        kwargs : dict
            Keyword args given to the plot function

        Returns
        -------
        Tuple of (List of Lines2D)
        """
        ds = self.pol_ds
        if not hasattr(self.pol_ds, "fit_exp_result_"):
            raise ValueError("The PolTRSpec must have successfully fit the " "data")
        if ax is None:
            ax = plt.gca()
        is_nm = self.freq_unit == "nm"

        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        f = ds.fit_exp_result_.fitter
        num_exp = f.num_exponentials
        taus = f.last_para[-num_exp:]
        leg_text = [ph.nsf(i) + " " + ph.time_unit for i in taus]
        if max(f.last_para) > 5 * f.t.max():
            leg_text[-1] = "const."
        n = ds.para.wavelengths.size
        x = ds.para.wavelengths if is_nm else ds.para.wavenumbers
        start = 0

        if any(np.diff(taus) < 0):
            raise ValueError("SAS assumes sorted taus")

        das = f.c[:, :num_exp]
        edas_pa = np.cumsum(das[:n, ::-1], axis=1)[:, ::-1]
        edas_pe = np.cumsum(das[n:, ::-1], axis=1)[:, ::-1]

        palines = []
        pelines = []

        for c, i in enumerate(range(start, num_exp)):
            l1 = ax.plot(x,
                         edas_pa.T[i],
                         **kwargs,
                         **self.para_ls,
                         label=leg_text[i],
                         color='C%d' % c)
            l2 = ax.plot(x, edas_pe.T[i], **kwargs, **self.perp_ls)
            dv.equal_color(l1, l2)
            palines += l1
            pelines += l2
        ph.lbl_spec(ax=ax)
        ncol = max(num_exp // 3, 1)
        if add_legend:
            ax.legend(title="SAS\nConstants", ncol=ncol)
        return palines, pelines

    def trans_anisotropy(self, wls, symlog=True, ax=None, freq_unit="auto"):
        """
        Plots the anisotropy over time for given frequencies.
        Parameters
        ----------
        wls : list of floats
            Which frequencies are plotted.
        symlog : bool
            Use symlog scale
        ax : plt.Axes or None
            Matplotlib Axes, if `None`, defaults to `plt.gca()`.

        Returns
        -------
        : list of Line2D
            List with the line objects.

        """
        if ax is None:
            ax = plt.gca()
        ds = self.pol_ds
        tmp = self.freq_unit if freq_unit == "auto" else freq_unit
        is_nm = tmp == "nm"
        x = ds.wavelengths if is_nm else ds.wavenumbers
        if is_nm:
            ph.vis_mode()
        else:
            ph.ir_mode()
        l = []
        for i in wls:
            idx = dv.fi(x, i)
            pa, pe = ds.para.data[:, idx], ds.perp.data[:, idx]
            aniso = (pa-pe) / (2*pe + pa)
            l += ax.plot(ds.para.t,
                         aniso,
                         label=ph.time_formatter(ds.t[idx], ph.time_unit))
        ph.lbl_trans(use_symlog=symlog)
        if symlog:
            ax.set_xscale("symlog")
        ax.set_xlim(-1)
        return l


class DataSetInteractiveViewer:
    def __init__(self, dataset, fig_kws=None):
        """
        Class showing a interactive matplotlib window for exploring
        a dataset.
        """
        if fig_kws is None:
            fig_kws = {}

        self.dataset = ds = dataset
        self.figure, axs = plt.subplots(3, 1, **fig_kws)
        self.ax_img, self.ax_trans, self.ax_spec = axs
        self.ax_img.pcolormesh(dataset.wn, dataset.t, dataset.data)
        self.ax_img.set_yscale("symlog", linscale=1)
        ph.lbl_spec(self.ax_spec)
        ph.lbl_trans(self.ax_trans)
        self.ax_trans.set_xscale('symlog', linthresh=1)
        
        self.trans_line = self.ax_trans.plot([])[0]
        self.spec_line = self.ax_spec.plot([])[0]
        self.ax_trans.set_xlim(-1, ds.t.max())
        self.ax_trans.set_ylim(ds.data.min(), ds.data.max())
        self.ax_spec.set_ylim(ds.data.min(), ds.data.max())
        self.ax_spec.set_xlim(ds.wn.max(), ds.wn.min())
        
        self._events = []
        self.init_events()

    
    def init_events(self):
        """Connect mpl events"""
        
        connect = self.figure.canvas.mpl_connect
        self._events.append(connect("motion_notify_event", self.update_lines))

    def update_lines(self, event):
        """If the mouse cursor is over the 2D image, update
        the dynamic transient and spectrum"""
        print(event.inaxes)
        self.ax_img.set_title(self.ax_spec)
        if event.inaxes is self.ax_img:
            ds = self.dataset
            print(event)
            wn_idx = ds.wn_idx(event.xdata)
            t_idx = ds.t_idx(event.ydata)
            spec = ds.data[t_idx, :]
            trans = ds.data[:, wn_idx]
            self.ax_trans.set_title('%.1f'%ds.wn[wn_idx])
            self.ax_trans.set_ylim(trans.min(), trans.max())
            self.ax_spec.set_title('%.1f'%ds.t[t_idx])
            self.ax_spec.set_ylim(spec.min(), spec.max())
            
            self.trans_line.set_data(ds.t, ds.data[:, wn_idx])        
            self.spec_line.set_data(ds.wn, ds.data[t_idx, :])
                        
            self.figure.canvas.draw()

@attr.s(auto_attribs=True)
class SasViewer:
    fit_result : FitExpResult
    model : Model 
    fig : plt.Figure
    lines : List[Line2D]

    