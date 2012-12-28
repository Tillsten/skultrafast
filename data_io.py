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
    
    num_list = [i[i.find('dat') - 4 - 1] for i in files]
    print num_list
    endname = max(zip(map(int, num_list), files))[1]
    print 'Loading: ' + endname
    a = np.load(endname)
    files = glob.glob(name + '-???_?_' + '*dat.npy')
    print files
    wls = []
    for i in files:
        print 'Loading: ' + i
        tmp = np.load(i)
        t, w = tmp[1:,0], tmp[0,1:]
        wls.append(w)
    return t, wls, a

def concate_data(wls,dat):   
    """ Puts the data from different central wavelengths into one Array"""
    
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
    spec = f.c.T[:, :-4] if f.model_coh else f.c.T
    arr = np.column_stack((f.wl, spec))
    taus = np.hstack((0, f.last_para[2:]))
    
    arr = np.vstack((taus, arr))
    np.savetxt(name, arr)

def save_txt(name, wls, t, dat):
    try:
        tmp = np.vstack((wls[None, :], dat))
        arr = np.hstack((np.vstack((0,t[:,None])), tmp))
    except ValueError:
        print wls.shape, t. shape, dat.shape
        raise IndexError
    np.savetxt(name, arr)
    

