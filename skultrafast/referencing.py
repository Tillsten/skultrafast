# This file contains function to apply adavanced referencing to raw dataset

import h5py
import numpy as np
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from attr import dataclass


def get_stats(grp: h5py.Group) -> np.ndarray:
    keys = list(grp.keys())
    n = len(keys)
    dim = grp[keys[0]].shape
    out = np.zeros((n, *dim), dtype=np.float32)
    for i in range(n):
        out[i, :] = grp[str(i)][:].astype(np.float32)
    return out


def get_scans(f: h5py.File) -> dict[str, list[np.ndarray]]:
    out = defaultdict(list)
    for p in 'Probe1', 'Probe2':
        for i in range(len(f['frames'][p])):
            out[p].append(get_stats(f['frames'][p][str(i)]))
    return out


def get_ref_stats(f: h5py.File) -> list[np.ndarray]:
    out = []
    for i in range(len(f['ref_data'])):
        out.append(get_stats(f['ref_data'][str(i)]))
    return out


def get_all_scans(f: h5py.File, filter_val=0.8) -> np.ndarray:
    data = get_scans(f)
    max_scan = min([arr.shape[0] for arr in data['Probe1']])
    _, n_wl, num_frames = data['Probe1'][0].shape
    n_t2 = len(data['Probe1'])
    if 'ref_data' in f:
        has_ref = f['ref_data/0/0'].shape[1] == num_frames
        ref_data = get_ref_stats(f)
    else:
        has_ref = False
        ref_data = None
    lines = 3 if has_ref else 2

    all_scans = np.empty((lines, max_scan, n_t2, n_wl, num_frames))
    for i in range(n_t2):
        all_scans[0, :, i, ...] = data['Probe1'][i][:max_scan, ...]
        all_scans[1, :, i, ...] = data['Probe2'][i][:max_scan, ...]
        if has_ref:
            assert ref_data is not None
            all_scans[2, :, i, ...] = ref_data[i][:max_scan, ...]
    if filter_val > 0:
        all_scans = gaussian_filter1d(all_scans, filter_val, axis=-2)
    return all_scans


def build_basis(all_scans):
    pass


def use_edge_ref(probe_wn, all_scan, low, high, ref_scans=2):
    """
    This function takes the whole 2D dataset and applies edge referencing to it.
    After referencing, the 2D-signal is calculated and returned along with the
    corresponding wavenumbers.

    Assumes 4 phase cycling steps.
    """

    data_idx = (probe_wn < high) & (probe_wn > low)
    data = all_scan[:, :, :, data_idx, :]
    edge = all_scan[:, :, :, ~data_idx, :]
    data_diffs = np.diff(data, axis=-1)

    edge_diffs = np.diff(edge, axis=-1)
    num_frames = data.shape[-1]
    out = np.empty_like(data_diffs)[..., ::4]
    max_scan = data.shape[1]
    for k in [0, 1]:
        for i in range(data.shape[1]):
            zero0 = np.concatenate(data_diffs[k, i, :ref_scans, ...], axis=-1)
            zero_base0 = np.concatenate(edge_diffs[k, i, :ref_scans, ...], axis=-1)
            if i != max_scan - 1:
                zero_tmp = np.concatenate(data_diffs[k, i+1, :ref_scans, ...], axis=-1)
                zero_base_tmp = np.concatenate(
                    edge_diffs[k, i+1, :ref_scans, ...], axis=-1)
                zero0 = np.concatenate((zero0, zero_tmp), axis=-1)
                zero_base0 = np.concatenate((zero_base0, zero_base_tmp), axis=-1)

            coeffs = np.linalg.lstsq(zero_base0.T, zero0.T, rcond=None)[0]
            for j in range(ref_scans, data_diffs.shape[2]):
                sig = data_diffs[k, i, j, ...]
                means = data[k, i, j, ...].mean(axis=-1)
                reffed = (edge_diffs[k, i, j, ...].T @ coeffs).T
                sig = (sig - reffed)[:, ::2]
                sig = (sig[:, :-1:2] + sig[:, 1::2])/means[:, None]
                fac = -1000/np.log(10)
                sig = np.log1p(sig)*fac
                sig[..., 0] *= 0.5
                sig *= np.hamming(num_frames/2)[sig.shape[1]:]
                out[k, i, j, ...] = sig
    out2d = np.fft.rfft(out.mean(1), axis=-1, n=2*num_frames).real
    return out2d, probe_wn[data_idx]


def use_edge_ref_full(all_scan, probe_wn, low, high, add_higher_degree):
    """
    This function takes the whole 2D dataset and applies edge referencing to it.
    After referencing, the 2D-signal is calculated and returned along with the
    corresponding wavenumbers. Assumes 4 phase cycling steps.


    Parameters
    ----------
    all_scan : np.ndarray
        The 2D dataset.
    probe_wn : np.ndarray
        The wavenumbers of the dataset.
    low : float
        The lower bound of the wavenumber range to exclude.
    high : float
        The upper bound of the wavenumber range to exclude.
    add_higher_degree : bool
        Whether to add higher degree polynomials to the referencing.
    """

    data_idx = (probe_wn < high) & (probe_wn > low)
    data = all_scan[:, :, :, data_idx, :]
    edge = all_scan[:, :, :, ~data_idx, :]
    data_diffs = np.diff(data, axis=-1)
    edge_diffs = np.diff(edge, axis=-1)
    num_frames = data.shape[-1]

    zero_a = np.concatenate(data_diffs[0, :, :2, ...], axis=-1)
    zero_a = np.concatenate(zero_a, axis=-1)
    zero_b = np.concatenate(data_diffs[1, :, :2, ...], axis=-1)
    zero_b = np.concatenate(zero_b, axis=-1)
    zero_base_a = np.concatenate(edge_diffs[0, :, :2, ...], axis=-1)
    zero_base_a = np.concatenate(zero_base_a, axis=-1)
    zero_base_b = np.concatenate(edge_diffs[1, :, :2, ...], axis=-1)
    zero_base_b = np.concatenate(zero_base_b, axis=-1)

    if add_higher_degree:
        zero_base_a = np.vstack((zero_base_a, zero_base_a**2))
        zero_base_b = np.vstack((zero_base_b, zero_base_b**2))
        print(zero_base_a.shape, zero_base_b.shape)
    coefs_a = np.linalg.lstsq(zero_base_a.T, zero_a.T, rcond=None)[0]
    coefs_b = np.linalg.lstsq(zero_base_b.T, zero_b.T, rcond=None)[0]

    sig_a = data_diffs.mean(1)[0]
    base = np.swapaxes(edge_diffs[0].mean(0), -1, -2)

    if add_higher_degree:
        base = np.dstack((base, base**2))

    reffed_a = (base @ coefs_a).swapaxes(-1, -2)

    sig_b = data_diffs.mean(1)[1]

    base = np.swapaxes(edge_diffs[1].mean(0), -1, -2)
    if add_higher_degree:
        base = np.dstack((base, base**2))
    reffed_b = (base @ coefs_b).swapaxes(-1, -2)

    sig = np.stack((sig_a, sig_b), axis=0)
    reffed = np.stack((reffed_a, reffed_b), axis=0)
    sig.shape, reffed.shape

    means = data[:2].mean(1).mean(-1)
    sig = (sig - reffed)[..., ::2]
    # If not using 4 phase cycling steps, change the indexing here
    sig = (sig[..., :-1:2] + sig[..., 1::2])/means[..., None]
    fac = 1000/np.log(10)
    sig = np.log1p(sig)*fac
    sig[..., 0] *= 0.5
    sig *= np.hamming(num_frames/2)[sig.shape[-1]:]

    out2d = np.fft.rfft(sig, axis=-1, n=4*num_frames//4).real
    return out2d, probe_wn[data_idx]
