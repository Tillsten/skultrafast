# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:34:30 2012

@author: tillsten
"""

import numpy as np

def vbload(fname = 'C:\Users\Tillsten\Documents\weisslicht.dat'):
    """
    loads a old vb file
    """
    raw = load_datfile(fname)
    (times, wl, sig, se, ref, re) = read_data(raw)
    return  (times, wl, sig, se, ref, re)

def load_datfile(datfile):
    """Load old vb-data file """
    f = lambda  s: float(s.replace(',', '.'))
    d = np.loadtxt(datfile, converters = {0:f, 1:f, 2:f, 3:f})
    return d    
    
def read_data(d):    
    """
    Put raw data into arrays.
    """
    num_times = len(d) / (800)
    times = np.zeros((num_times))
    wl = np.zeros((400))#
    for i in range(0, 400):
        wl[i] = d[i, 1]
    sig = np.zeros((num_times, 400))
    ref = np.zeros((num_times, 400))
    sig_err = np.zeros((num_times, 400))
    ref_err = np.zeros((num_times, 400))
    for i in range(num_times):
        times[i] = d[i * 800 + 1, 0]
        for j in range(0, 400):
            sig[i, j] = d[i * 800 + j, 2]
            sig_err[i, j] = d[i * 800 + j, 3]
            ref[i, j] = d[i * 800 + j + 400, 2]
            ref_err[i, j] = d[i * 800 + j + 400, 3]
    return (times, wl, sig, sig_err, ref, ref_err)

def loader_func(name):
    """ Helper function to load data from MessPy"""
    import glob
    
    files = glob.glob(name + '_dat?.npy') + glob.glob(name + '_dat??.npy')
    if len(files) == 0:
        raise IOError('No file found.')
    print files
    import re    
    num_list = [re.findall('dat\d+', i)[0][3:] for i in files]    
    
    endname = max(zip(map(int, num_list), files))[1]
    
    a = np.load(endname)
    num_list = [str(int(i) - 1) for i in num_list]
    print num_list
    num = str(max(map(int, num_list)))
    print num
    print name
    search_string = name + '-???_'+ '*' + '_' + 'dat.npy'
    print search_string
    files = glob.glob(search_string)
    wls = []
    print files
    for i in files:
        print 'Loading: ' + i
        cwl = re.findall('-\d\d\d_', i)
        tmp = np.load(i)
        t, w = tmp[1:,0], tmp[0,1:]
        wls.append(w)
    return t, wls, a

def concate_data(wls,dat):   
    """Puts the data from different central wavelengths into one Array"""
    
    w = np.hstack(tuple(wls))
    idx = np.argsort(w)
    w = w[idx]
    k = dat.shape
    #Remove wl index, it a more fancy way to write a columnstack.
    dat = dat.reshape(k[0], k[1]*k[2], k[3], order='F')
    dat = dat[:, idx, :]
    return w, dat

def concate_data_pol(wls, dat):
    w = np.hstack(tuple(wls.T))
    idx = np.argsort(w)    
    w = w[idx]    
    k = dat.shape    
    dat =np.hstack(tuple(dat))
    #Remove wl index, it a more fancy way to write a columnstack.
    dat = dat[:, idx, :]
    return w, dat

def save_txt_das(name, fitter):
    """
    Saves the das of a fitter-obj in name.txt
    """
    f = fitter
    spec = f.c[:, :-4] if f.model_coh else f.c
    arr = np.column_stack((f.wl, spec))
    offset = f.model_disp+1
    taus = np.hstack((0, f.last_para[offset:]))
    
    arr = np.vstack((taus, arr))
    np.savetxt(name, arr)

def make_report(fitter, info, raw=None, plot_fastest=1):
    import plot_funcs
    g = fitter
    name = info.get('name','')
    solvent = info.get('solvent','')    
    excitation = info.get('excitation','')
    add_info = info.get('add_info', '')
    title = u"{} in {} excited at {}. {}".format(name, solvent, excitation, add_info)
    plot_funcs.a4_overview(g, 'pics\\' + title + '.png', title=title, 
                           plot_fastest=plot_fastest)
    save_txt_das(name + '_-DAS.txt', g)
    save_txt(name + '_ex' + excitation + '_data.txt', g.wl, g.t, g.data)
    save_txt(name + '_ex' + excitation + '_fit.txt', g.wl, g.t, g.model)
    if raw:
        save_txt(name +  '_ex' + excitation + '_raw.txt', *raw)

def save_txt(name, wls, t, dat):
    try:
        tmp = np.vstack((wls[None, :], dat))
        arr = np.hstack((np.vstack((0,t[:,None])), tmp))
    except ValueError:
        print wls.shape, t.shape, dat.shape
        raise IndexError
    np.savetxt(name, arr)

def svd_filter(d, n=10):
    u, s, v = np.linalg.svd(d, full_matrices=False)
    s[n:] = 0.
    return np.dot(u, np.dot(np.diag(s), v))

import numba 

def sort_scans(data):
    axis = -1
    index = list(np.ix_(*[np.arange(i) for i in data.shape]))
    index[axis] = np.abs(data-data.mean(-1)[..., None]).argsort(axis)
    dsorted = data[index]
    return dsorted

import re
def extract_freqs_from_gaussianlog(fname):
    f = open(fname)
    fr, ir, raman = [], [], []
    
    for line in f:
        if line.lstrip().startswith('Frequencies ---'):
            fr += (map(float, re.sub(r'[^\d,. ]', '', line).split()))
        elif line.lstrip().startswith('IR Intensi'):
            ir+=(map(float, re.sub(r'[^\d,. ]', '', line).split()))
        elif line.lstrip().startswith('Raman Activities'):
            raman += (map(float, re.sub(r'[^\d,. ]', '', line).split()))
    arrs = map(np.array, [fr, ir, raman])
    return np.vstack([i.flatten() for i in arrs])
    

