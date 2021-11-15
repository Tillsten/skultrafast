# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:34:30 2012

@author: tillsten
"""
from __future__ import print_function
import pathlib
import numpy as np
import json
from pathlib import Path
import urllib.request
import re
import hashlib
import zipfile
import zipfile_deflate64
import os, subprocess

def vbload(fname=r'C:\Users\Tillsten\Documents\weisslicht.dat'):
    """
    loads a old vb file
    """
    raw = load_datfile(fname)
    (times, wl, sig, se, ref, re) = read_data(raw)
    return (times, wl, sig, se, ref, re)


def load_datfile(datfile):
    """Load old vb-data file """
    f = lambda s: float(s.replace(',', '.'))
    d = np.loadtxt(datfile, converters={0: f, 1: f, 2: f, 3: f})
    return d


def read_data(d):
    """
    Put raw data into arrays.
    """
    num_times = len(d) / (800)
    times = np.zeros((num_times))
    wl = np.zeros((400))  #
    for i in range(0, 400):
        wl[i] = d[i, 1]
    sig = np.zeros((num_times, 400))
    ref = np.zeros((num_times, 400))
    sig_err = np.zeros((num_times, 400))
    ref_err = np.zeros((num_times, 400))
    for i in range(num_times):
        times[i] = d[i*800 + 1, 0]
        for j in range(0, 400):
            sig[i, j] = d[i*800 + j, 2]
            sig_err[i, j] = d[i*800 + j, 3]
            ref[i, j] = d[i*800 + j + 400, 2]
            ref_err[i, j] = d[i*800 + j + 400, 3]
    return (times, wl, sig, sig_err, ref, ref_err)


def loader_func(name):
    """ Helper function to load data from MessPy"""
    import glob

    files = glob.glob(name + '_dat?.npy') + glob.glob(name + '_dat??.npy')
    if len(files) == 0:
        raise IOError('No file found.')

    import re
    num_list = [re.findall(r'dat\d+', i)[0][3:] for i in files]

    endname = max(zip(map(int, num_list), files))[1]

    a = np.load(endname)
    num_list = [str(int(i) - 1) for i in num_list]

    num = str(max(map(int, num_list)))

    search_string = name + '-???_' + '*' + '_' + 'dat.npy'

    files = glob.glob(search_string)
    wls = []

    for i in files:

        cwl = re.findall(r'-\d\d\d_', i)
        tmp = np.load(i)
        t, w = tmp[1:, 0], tmp[0, 1:]
        wls.append(w)
    return t, wls, a


def concate_data(wls, dat):
    """Puts the data from different central wavelengths into one Array"""

    w = np.hstack(tuple(wls))
    idx = np.argsort(w)
    w = w[idx]
    k = dat.shape
    #Remove wl index, it a more fancy way to write a columnstack.
    dat = dat.reshape(k[0], k[1] * k[2], k[3], order='F')
    dat = dat[:, idx, :]
    return w, dat


def concate_data_pol(wls, dat):
    w = np.hstack(tuple(wls.T))
    idx = np.argsort(w)
    w = w[idx]
    k = dat.shape
    dat = np.hstack(tuple(dat))
    #Remove wl index, it a more fancy way to write a columnstack.
    dat = dat[:, idx, :]
    return w, dat


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


def make_report(fitter, info, raw=None, plot_fastest=1, make_ltm=False):

    from skultrafast import zero_finding, dv, plot_funcs

    g = fitter
    name = info.get('name', '')
    solvent = info.get('solvent', '')
    excitation = info.get('excitation', '')
    add_info = info.get('add_info', '')
    title = u"{} in {} excited at {}. {}".format(name, solvent, excitation, add_info)
    plot_funcs.a4_overview(g,
                           'pics\\' + title + '.png',
                           title=title,
                           plot_fastest=plot_fastest)

    save_txt_das(name + '_-DAS.txt', g)
    save_txt(name + '_ex' + excitation + '_iso.txt', g.wl, g.t, g.data)
    save_txt(name + '_ex' + excitation + '_iso_fit.txt', g.wl, g.t, g.model)

    dat = zero_finding.interpol(dv.tup(fitter.wl, fitter.t, fitter.data), fitter.tn, 0.0)
    save_txt(name + '_ex' + excitation + '_iso_timecor.txt', *dat)
    if make_ltm:
        plot_funcs.plot_ltm_page(dat, 'pics\\' + title + 'lft_map.png')

    fit = zero_finding.interpol(dv.tup(fitter.wl, fitter.t, fitter.model), fitter.tn, 0.0)
    save_txt(name + '_ex' + excitation + '_iso_fit_timecor.txt', *fit)

    if raw:
        save_txt(name + '_ex' + excitation + '_raw.txt', *raw)

    if hasattr(fitter, 'data_perp'):
        perp = zero_finding.interpol(dv.tup(fitter.wl, fitter.t, fitter.data_perp),
                                     fitter.tn, 0.0)

        para = zero_finding.interpol(dv.tup(fitter.wl, fitter.t, fitter.data_para),
                                     fitter.tn, 0.0)

        #plot_funcs.a4_overview_second_page(fitter, para, perp, 'bla.png')
        save_txt(name + '_ex' + excitation + '_para.txt', *para)
        save_txt(name + '_ex' + excitation + '_perp.txt', *perp)
    import matplotlib.pyplot as plt
    plt.close('all')


def save_txt(name, wls, t, dat, fmt='%.3f'):
    try:
        tmp = np.vstack((wls[None, :], dat))
        arr = np.hstack((np.vstack((0, t[:, None])), tmp))
    except ValueError:
        print('Shapes wl:', wls.shape, 't', t.shape, 'd', dat.shape)
        raise IndexError
    np.savetxt(name, arr, fmt=fmt)


def svd_filter(d, n=10):
    u, s, v = np.linalg.svd(d, full_matrices=False)
    s[n:] = 0.
    return np.dot(u, np.dot(np.diag(s), v))


def sort_scans(data):
    axis = -1
    index = list(np.ix_(*[np.arange(i) for i in data.shape]))
    index[axis] = np.abs(data - data.mean(-1)[..., None]).argsort(axis)
    dsorted = data[index]
    return dsorted


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
    data_dir =  (p / 'MeSCN_2D_data')
    if not data_dir.exists():
        if os.name == 'nt':
            # Somehow the file only works under windows
            zipfile_deflate64.ZipFile((p / 'MeSCN_2D_data.zip')).extractall(p / 'MeSCN_2D_data')
        else:
            cmd = 'unzip %s -d %s' %(p / 'MeSCN_2D_data.zip', p / 'MeSCN_2D_data/')
            output = os.popen(cmd).read()
    if len(list(data_dir.glob('*.*'))) != 1106:
        raise IOError("Extract failed")
    return p / 'MeSCN_2D_data'
