# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:34:30 2012

@author: tillsten
"""
from __future__ import print_function

import hashlib
import json
import os
import pathlib
import re
import urllib.request
from pathlib import Path

import numpy as np
import zipfile_deflate64


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

            fr += (map(float, re.sub(r'[^\d,.\d\d\d\d]', '', line).split()))
        elif line.lstrip().startswith('IR Inten'):
            ir += (map(float, re.sub(r'[^\d,.d\d\d\d]', '', line).split()))
        elif line.lstrip().startswith('Raman Activities'):
            raman += (map(float, re.sub(r'[^\d,.\d\d\d\d ]', '', line).split()))
    arrs = map(np.array, [fr, ir, raman])
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


def get_twodim_dataset():
    """Checks if the two-dim example data is in the skultrafast folder,
       if not, downloads it from figshare. Returns the path to file zip-file"""
    p = pathlib.Path.home() / 'skultrafast_data'
    if not p.exists():
        p.mkdir()
    if len(list(p.glob('MeSCN_2D_data.zip'))) == 0:
        article_id = 15156528
        ans = urllib.request.urlopen(
            f'https://api.figshare.com/v2/articles/{article_id}/files')
        if ans.status == 200:
            d = json.loads(ans.read())[0]
            name = d['name']
            durl = d['download_url']
        else:
            raise IOError("Downloading example data filed")
        ans = urllib.request.urlopen(durl)
        if ans.status == 200:
            content = ans.read()
            md5 = hashlib.md5(content)
            if md5.hexdigest() == 'daa0562d3d0f1518f3e2952d98082591':
                with (p / 'MeSCN_2D_data.zip').open('wb') as f:
                    f.write(content)
            else:
                raise IOError('MD5-hash of download not correct')
        else:
            raise IOError("Figshare ans != 200, %s instead" % ans.status)
    data_dir = (p / 'MeSCN_2D_data')
    if not data_dir.exists():
        if os.name == 'nt':
            # Somehow the file only works under windows
            zipfile_deflate64.ZipFile((p / 'MeSCN_2D_data.zip')
                                      ).extractall(p / 'MeSCN_2D_data')
        else:
            cmd = 'unzip %s -d %s' % (p / 'MeSCN_2D_data.zip', p / 'MeSCN_2D_data/')
            output = os.popen(cmd).read()
    if len(list(data_dir.glob('*.*'))) != 1106:
        raise IOError("Extract failed")
    return p / 'MeSCN_2D_data'
