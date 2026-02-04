from copy import deepcopy
import json
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Tuple,
    Union,
    no_type_check,
    TYPE_CHECKING,
)

import attr
import h5py
from matplotlib.dates import TH
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from skultrafast.twoD_dataset import TwoDim
from skultrafast.unit_conversions import THz2cm
from skultrafast.utils import poly_bg_correction, sigma_clip

if TYPE_CHECKING:
    import matplotlib.axes


def _compute_means_and_stderr(t2_index, ifr, use_clip) -> Tuple[dict, dict]:
    """Compute the means and standard errors of the interferograms at a given
    t2 index. If `use_clip` is True, use sigma clipping to filter outliers.
    Not part of class to be used with joblib.

    Parameters
    ----------
    t2_index : int
        The index of the t2 array to use.
    ifr : dict
        Dictionary containing the interferogram data.
    use_clip : bool
        If True, use sigma clipping to filter outliers.

    """
    means = {}
    stderr = {}
    for name in ifr:
        if not use_clip:
            m = np.mean(ifr[name][str(t2_index)], 0)
            s = np.std(ifr[name][str(t2_index)], 0)
        else:
            masked_arr, m, s = sigma_clip(
                ifr[name][str(t2_index)],
                axis=0,
                max_iter=4,
                sigma=2.5,
                full_return=True,
            )
            m = m.data[0, ...]
            s = s.data[0, ...]
        means[name] = m
        stderr[name] = s
    return means, stderr


@attr.define
class Messpy25File:
    """
    Class for working with MessPy2D 2D-IR files.
    """

    h5_file: Union[str, Path] = attr.ib(init=True)
    "h5py file object containing the 2D dataset, the only required parameter"
    is_para_array: Literal["Probe1", "Probe2"] = "Probe1"
    "which dataset has parallel polarisation"
    probe_wn: np.ndarray = attr.ib(init=False)
    "Array with probe wavenumbers"
    pump_wn: np.ndarray = attr.ib(init=False)
    "Array with the pump wavenumbers. Depends on the upsampling used during measurment"
    t2: np.ndarray = attr.ib(init=False)
    "Array containing the waiting times"
    t1: np.ndarray = attr.ib(init=False)
    "Array containing the double pulse delays"
    rot_frame: float = attr.ib(init=False)
    "Rotation frame used while measuring"
    meta: dict = attr.ib(init=False, factory=dict)
    "Metadata stored in the file"
    phase_cycles: int = attr.ib(init=False, default=4)
    "Number of phase cycles used during the measurement"
    THREADING_ENABLED: bool = False
    "If true, use threading for parallel processing"

    @no_type_check
    def __attrs_post_init__(self):
        if isinstance(self.h5_file, h5py.File):
            raise ValueError(
                "Directly passing an h5py.File is not supported "
                "anymore, please pass a path to the file instead."
            )

        with h5py.File(self.h5_file, "r") as f:
            if "t1" in f:
                # Old bug in naming of t1 and t2
                self.t1 = f["t1"][:]
                self.t2 = f["t2"][:]
            else:
                self.t1 = f["t2"][:]
                self.t2 = f["t3"][:]
            self.rot_frame = f["t1"].attrs["rot_frame"]
            self.probe_wn = f["wn"][:]
            i: np.ndarray = f["ifr_data/Probe1/0/0"]
            self.pump_wn = THz2cm(
                np.fft.rfftfreq(2 * i.shape[1], (self.t1[1] - self.t1[0]))
            )

            self.pump_wn += self.rot_frame
            if "meta" in f:
                self.meta = json.loads(f["meta"].attrs["meta"])

    def normalization_factor(self, n_phi=2) -> float:
        """
        Return the factor to turn the FFT out
        into a proper 2D spectrum.
        """
        dt1 = self.t1[1] - self.t1[0]  # in p
        N = self.t1.size
        factor = np.sqrt(dt1 / N) * np.sqrt(3e-5) * 2 / n_phi
        return factor

    def get_means(self):
        with h5py.File(self.h5_file, "r") as f:
            if not "2d_data" in f:
                raise ValueError("No 2D data found in file")
            means = {}
            for name, l in f["2d_data"].items():
                means[name] = []
                for i in range(self.t2.size):
                    means[name].append(l[str(i)]["mean"])
        para = self.is_para_array

        perp = "Probe2" if self.is_para_array == "Probe1" else "Probe2"
        para_means = np.stack(means[para], 0)
        perp_means = np.stack(means[perp], 0)
        return para_means, perp_means, 2 / 3 * perp_means + 1 / 3 * para_means

    def get_all_ifr(
        self, scan_selection: Optional[Any] = None
    ) -> dict[str, dict[str, list[np.ndarray]]]:
        """
        Retrieves and organizes interferogram (ifr) data from an HDF5 file.

        This method iterates over the 'ifr_data' group in the HDF5 file, and for
        each item, it creates a nested dictionary structure in Python. The outer
        dictionary's keys are the names of the items in 'ifr_data', and the
        values are inner dictionaries. The inner dictionaries' keys are the
        string representations of the indices in the range of the size of
        'self.t2', and the values are lists of data from the datasets in the
        HDF5 file.

        Returns:
            dict: A nested dictionary containing the organized ifr data.
        """
        ifr = {}
        with h5py.File(self.h5_file, "r") as f:
            for name, l in f["ifr_data"].items():
                ifr[name] = {}
                for i in range(self.t2.size):
                    li = []
                    scans = [int(s) for s in l[str(i)].keys() if s != "mean"]
                    if scan_selection is not None:
                        scans = [s for s in scans if s in list(scan_selection)]
                    scans.sort()
                    for scan in scans:
                        li.append(f[f"ifr_data/{name}/{str(i)}/{scan}"][:])
                    ifr[name][str(i)] = li
        return ifr

    def ifr_means_and_stderr(self, use_clip=True, scan_selection=None):
        ifr = self.get_all_ifr(scan_selection)
        if self.THREADING_ENABLED:
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(_compute_means_and_stderr)(t2_index, ifr, use_clip)
                for t2_index in range(self.t2.size)
            )
        else:
            results = [
                _compute_means_and_stderr(t2_index, ifr, use_clip)
                for t2_index in range(self.t2.size)
            ]
        means = {name: [] for name in ifr}
        stderr = {name: [] for name in ifr}
        for mean, err in results:
            for name in mean:
                means[name].append(mean[name])
                stderr[name].append(err[name])
        return means, stderr

    def get_ifr(
        self,
        probe_filter: None | float | tuple[int, int] = None,
        bg_correct=None,
        ch_shift: int = 0,
        use_clip=True,
        scan_selection=None,
        is_para_array: Literal["Probe1", "Probe2"] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the interferograms. If necessary, apply probefilter and
        background correction.

        Parameters
        ----------
        probe_filter: float or tuple or None
            The probe filter width in channels. (Gaussian filter) If a tuple of
            length two is given, apply savgol filter with the given parameters
            window and polynomial degree. If None, no filter is applied.
        bg_correct: Tuple[int, int]
            Number of left and right channels to use for background correction.
        ch_shift: int
            Number of o shift the Probe2 data. Corrects for
            missaligned channels.
        use_clip: bool
            If true, use sigma clipping to filter outliers.
        scan_selection: list or None
            If not None, only use the scans in the list.
        is_para_array: Literal["Probe1", "Probe2"]
            Which array is the parallel polarisation. If Probe1, Probe1 is
            parallel, else Probe2 is parallel. Overrides the
            `is_para_array` attribute of the class.
        Returns
        -------
        ifr: Tuple[np.ndarray, np.ndarray, np.ndarray]
            The interferograms for paralllel, perpendicular and isotropic
            polarisation. The shape of each array is (n_t2, n_probe_wn, n_t1).
        """

        means, stderr = self.ifr_means_and_stderr(
            use_clip=use_clip, scan_selection=scan_selection
        )
        if is_para_array is None:
            is_para_array = self.is_para_array

        para = is_para_array
        perp = "Probe2" if is_para_array == "Probe1" else "Probe1"

        para_means = np.stack(means[para], 0)
        perp_means = np.stack(means[perp], 0)
        if probe_filter is not None:
            if isinstance(probe_filter, tuple):
                para_means = savgol_filter(para_means, *probe_filter, axis=1)
                perp_means = savgol_filter(perp_means, *probe_filter, axis=1)
            elif isinstance(probe_filter, float):
                para_means = gaussian_filter1d(
                    para_means, probe_filter, 1, mode="nearest"
                )
                perp_means = gaussian_filter1d(
                    perp_means, probe_filter, 1, mode="nearest"
                )
        if ch_shift > 0:
            para_means = para_means[:, :-ch_shift, :]
            perp_means = perp_means[:, ch_shift:, :]
            wn = self.probe_wn[ch_shift:]
        elif ch_shift < 0:
            para_means = para_means[:, -ch_shift:, :]
            perp_means = perp_means[:, :ch_shift, :]
            wn = self.probe_wn[:ch_shift]
        else:
            wn = self.probe_wn
        if bg_correct is not None:
            for i in range(para_means.shape[0]):
                poly_bg_correction(wn, para_means[i].T, bg_correct[0], bg_correct[1])
                poly_bg_correction(wn, perp_means[i].T, bg_correct[0], bg_correct[1])
        iso_means = 2 / 3 * perp_means + 1 / 3 * para_means
        return para_means, perp_means, iso_means

    def make_two_d(
        self,
        upsample: int = 4,
        window_fcn: Optional[Callable] = np.hanning,
        ch_shift: int = 1,
        probe_filter: Optional[float] = None,
        bg_correct: Optional[Tuple[int, int]] = None,
        use_clip: bool = False,
        t0_factor: float = 0.5,
        scan_selection: Optional[list] = None,
        subtract_ifr_mean: bool = False,
        is_para_array: Literal["Probe1", "Probe2"] | None = None,
    ) -> Dict[str, TwoDim]:
        """
        Calculates the 2D spectra from the interferograms and returns it as a
        dictionary. The dictorary contains messpy 2D-objects for paralllel,
        perpendicular and isotropic polarisation.

        Parameters
        ----------
        upsample: int
            Upsampling factor used in the FFT. A factor over 2 only does sinc
            interpolation.
        window_fcn: Callable
            If given, apply a window function to the FFT.
        probe_filter: float
            The probe filter width in channels. (Gaussian filter)
        ch_shift: int
            Number of channels to shift the Probe2 data. Corrects for
            missaligned channels.
        bg_correct: Tuple[int, int]
            Number of left and right channels to use for background correction.
        use_clip: bool
            If true, use sigma clipping to filter outliers.
        t0_factor: float
            Factor to multiply the first t1 point (zero-delay between the pumps)
            to correct for the integration. In general, the default should not
            be touched.
        scan_selection: list or None
            If not None, only use the scans in the list. Useful for filtering
            out bad scans.
        subtract_ifr_mean: bool
            If True, subtract the mean of the interferograms before FFT.
        is_para_array: Literal["Probe1", "Probe2"]
            Which array is the parallel polarisation. If Probe1, Probe1 is
            parallel, else Probe2 is parallel. Overrides the `is_para_array`
            attribute of the class.
        """
        means = self.get_ifr(
            probe_filter=probe_filter,
            bg_correct=bg_correct,
            ch_shift=ch_shift,
            use_clip=use_clip,
            scan_selection=scan_selection,
            is_para_array=is_para_array,
        )
        data = {pol: means[i] for i, pol in enumerate(["para", "perp", "iso"])}
        out = {}
        for k, v in data.items():
            v[:, :, 0] *= t0_factor
            if window_fcn is not None:
                v = v * window_fcn(v.shape[2] * 2)[None, None, v.shape[2] :]
            if subtract_ifr_mean:
                v -= v.mean(2, keepdims=True)
            sig = np.fft.rfft(v, axis=2, n=v.shape[2] * upsample).real
            self.pump_wn = (
                THz2cm(
                    np.fft.rfftfreq(upsample * v.shape[2], (self.t1[1] - self.t1[0]))
                )
                + self.rot_frame
            )
            if ch_shift >= 0:
                probe_wn = self.probe_wn[ch_shift:]
            elif ch_shift < 0:
                probe_wn = self.probe_wn[:ch_shift]
            ds = TwoDim(self.t2, self.pump_wn, probe_wn, sig)
            ds.info = deepcopy(self.meta)
            out[k] = ds
        return out

    def get_scan_times(self, which_point="0") -> np.ndarray:
        """
        Returns a dictionary with the scan numbers and their creation times
        for the specified point in the interferogram data.

        Parameters
        ----------
        which_point : str
            The point in the interferogram data to retrieve scan times for.
            Default is '0', which corresponds to the first point in the t2 array.

        Returns
        -------
        np.ndarray
            An array of datetime64 objects representing the creation times of each scan.
        """
        scan_times = {}
        with h5py.File(self.h5_file, "r") as f:
            for scan in f["ifr_data"]["Probe1"][which_point]:
                if scan != "mean":
                    scan_times[int(scan)] = f["ifr_data"]["Probe1"]["0"][scan].attrs[
                        "creation date"
                    ]
        scan_times = sorted(scan_times)
        return np.array(scan_times.items(), dtype="datetime64")

    def get_detector_signals(self):
        """
        Returns the averaged detector signals for each scan, for both probes.
        """
        spectra = {}
        with h5py.File(self.h5_file, "r") as f:
            if "frames" not in f:
                raise ValueError(
                    "No frame data found in file."
                    "Frames are required to get detector signals."
                )
            for det in ["Probe1", "Probe2"]:
                spectra[det] = {}
                for scans in f["frames"][det]["0"]:
                    spectra[det][scans] = f["frames"][det]["0"][scans][:, :4].mean(
                        axis=1
                    )
        return spectra

    def get_single_probe_spectrum(
        self,
        scan: int,
        t2_index: int = 0,
        probe: Literal["Probe1", "Probe2"] = "Probe1",
    ) -> np.ndarray:
        with h5py.File(self.h5_file, "r") as f:
            if "frames" not in f:
                raise ValueError(
                    "No frame data found in file."
                    "Frames are required to get detector signals."
                )

            if f"frames/{probe}/{t2_index}/{scan}" not in f:
                raise ValueError(
                    f"Scan {scan} not found in {probe} for t2 index {t2_index}."
                )
            return f[f"frames/{probe}/{t2_index}/{scan}"][:]

    def make_model_fitfiles(self, path, name, probe_filter=None, bg_correct=None):
        """
        Saves the data in a format useable for the ModelFit Gui from Kevin Robben
        https://github.com/kevin-robben/model-fitting
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        ifr = self.get_ifr(probe_filter=probe_filter, bg_correct=bg_correct)
        data = {pol: ifr[i] for i, pol in enumerate(["para", "perp", "iso"])}
        idx = np.argsort(self.probe_wn)

        for pol in ["para", "perp", "iso"]:
            folder = p / pol
            folder.mkdir(parents=True, exist_ok=True)
            for i, t in enumerate(self.t2):
                fname = folder / (name + "_%f.txt" % t)
                d = data[pol][i, idx, :]

                np.savetxt(fname, d)
        np.savetxt(p / "pump_wn.txt", self.pump_wn)
        np.savetxt(p / "probe_wn.calib", self.probe_wn[idx])
        np.savetxt(p / "t1.txt", self.t1)
        np.savetxt(p / "t2.txt", self.t2)
        timestep = (self.t1[1] - self.t1[0]) * 1000

        np.savetxt(
            p / f"rot_frame_{self.rot_frame: .0f}_t1_stepfs_{timestep: .0f}.txt",
            [self.rot_frame],
        )

    def recalculate_wl(
        self, center_wl: float, center_ch: int = 65, disp: Optional[float] = None
    ):
        """
        Recalculates the wavelengths from the probe.
        """

        if disp is None:
            if np.diff(1e7 / self.probe_wn).max() < 6:
                disp = 7.8 / 2
            else:
                disp = 7.8
        wls = center_wl - disp * (np.arange(128) - center_ch)
        self.probe_wn = 1e7 / wls

    def _last_complete_scan(self) -> int:
        """
        Returns the index of the last complete scan, e.g. where all time points are present.
        """
        with h5py.File(self.h5_file, "r") as f:
            if "ifr_data" not in f:
                raise ValueError("No interferogram data found in file.")
            last_timepoint_grp = f["ifr_data"]["Probe1"][str(self.t2.size - 1)]
            scans = [int(scan) for scan in last_timepoint_grp if scan != "mean"]
            return max(scans) if scans else 0


@attr.define
class Messpy25Plotter:
    mp: Messpy25File

    def probe_spec(
        self,
        det: Literal["Probe1", "Probe2", "Ref"],
        scan_idx: int,
        t2_idx: int,
        ax: "None | matplotlib.axes.Axes" = None,
    ):
        """
        Plots the probe spectrum for a given scan index and t2 index.

        Parameters
        ----------
        scan_idx : int
            The index of the scan to plot.
        t2_idx : int
            The index of the t2 array to use.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, it uses the current axes.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()

        wn = self.mp.probe_wn
        data = self.mp.get_detector_signals()
        ax.plot(wn, data["Probe1"][str(t2_idx)][scan_idx].mean(axis=0))
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")
