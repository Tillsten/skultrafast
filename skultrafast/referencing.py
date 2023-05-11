import h5py
import numpy as np
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d


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
