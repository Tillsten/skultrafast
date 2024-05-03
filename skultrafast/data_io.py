# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:34:30 2012

@author: tillsten
"""
from __future__ import print_function


import re
from pathlib import Path
import pooch
import numpy as np


def save_txt_das(name, fitter):
    """
    Saves the das of a fitter-obj in name.txt
    """
    f = fitter
    spec = f.c[:, :-3] if f.model_coh else f.c
    arr = np.column_stack((f.wl, spec))
    offset = f.model_disp + 1
    taus = np.hstack((0, f.last_para[offset:]))

    arr = np.vstack((taus, arr))
    np.savetxt(name, arr)


def save_txt(name, wls, t, dat, fmt='%.3f'):
    try:
        tmp = np.vstack((wls[None, :], dat))
        arr = np.hstack((np.vstack((0, t[:, None])), tmp))
    except ValueError:
        print('Shapes wl:', wls.shape, 't', t.shape, 'd', dat.shape)
        raise IndexError
    np.savetxt(name, arr, fmt=fmt)


def extract_freqs_from_gaussianlog(fname):
    f = open(fname)
    fr, ir, raman = [], [], []

    for line in f:
        if line.lstrip().startswith('Frequencies --'):

            fr += (map(float, re.sub(r'[^\d,.\d\d\d\d]', ' ', line).split()))
        elif line.lstrip().startswith('IR Inten'):
            ir += (map(float, re.sub(r'[^\d,.d\d\d\d]', ' ', line).split()))
        elif line.lstrip().startswith('Raman Activities'):
            raman += (map(float, re.sub(r'[^\d,.\d\d\d\d ]', ' ', line).split()))

    arrs = [fr]
    if ir:
        arrs.append(ir)
    if raman:
        arrs.append(raman)

    arrs = map(np.array, [fr, ir])
    return np.vstack([i.flatten() for i in arrs])


def load_example():
    """
    Returns a tuple containing the example data shipped with skultrafast.

    Returns
    -------
    tuple of ndarrys
        Tuple with wavelengths, t and data-array.
    """
    import skultrafast
    a = np.load(skultrafast.__path__[0] + '/examples/data/test.npz')
    wl, data, t = a['wl'], a['data'], a['t']
    return wl, t*1000 - 2, data / 3.


def messpy_example_path():
    """
    Returns the path to the messpy example data shipped with skultrafast.

    Returns
    -------
    str
        The full path
    """
    import skultrafast
    return skultrafast.__path__[0] + '/examples/data/messpyv1_data.npz'


def get_example_path(kind):
    """Returns the path a example data-file.

    Parameters
    ----------
    kind : ('sys_response', 'messpy', 'vapor', 'ir_polyfilm', 'quickcontrol')
        Which path to return.
    """
    import skultrafast
    root = skultrafast.__path__[0] + '/examples/data/'
    file_dict = {
        "messpy": 'messpyv1_data.npz',
        "sys_response": 'germanium.npz',
        "vapor": 'ir_waterabs.npy',
        "ir_polyfilm": "PolystyreneFilm_spectrum.npz",
        "quickcontrol": "quickcontrol.zip"
    }
    return root + file_dict[kind]


POOCH = pooch.create(
    path=pooch.os_cache("skultrafast"),
    # Use the figshare DOI
    base_url="doi:10.6084/m9.figshare.25745715",
    registry={
        "MeSCN_2D_data.zip": "md5:6ca0942395a8b1be17b57a2b3c27ac5b",
    },
)


def get_twodim_dataset():
    data = POOCH.fetch("MeSCN_2D_data.zip", processor=pooch.Unzip())
    p = Path(data[0]).parent
    return p
