# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:14:29 2012

@author: tillsten
"""

from dv import interpol, tup, find_time_zero
from data_io import concate_data_pol, save_txt
import numpy as np
from scipy import ndimage as nd

try:
    t
except NameError:
    a = np.load('los5.npz')  
    wl = a['wl']
    t = a['t']/1000.
    d = a['data']
    
    wls, dat = concate_data_pol(wl, d[...,:4].mean(-1))
    dat = dat - dat[:30,:,:].mean(0)    
    ani = dat[:,:,0] + 2*dat[:,:,1]
    o = tup(wls, t, ani)

    tn, m = find_time_zero(o, 10)
    dn = interpol(ani, t, tn, 0, t)
    p = -interpol(dat[:,:,0], t, tn, 0, t)
    pf = nd.gaussian_filter(p,(0,3))
    s = -interpol(dat[:,:,1], t, tn, 0, t)
    sf = nd.gaussian_filter(s,(0,3))
def svd_zero(d, t, p):
    nd = interpol(d, t, 0, 0, t)
    u,s,v = np.svd(nd)
    return sum(s[:10])
    
    
save_txt('Alcor-py2-exec-620-13.12.12-para', wls, t, pf)
save_txt('Alcor-py2-exec-620-13.12.12-senk', wls, t, sf)
save_txt('Alcor-py2-exec-620-13.12.12-ani', wls, t, (2*sf+pf)/3.)

from fitter import Fitter 
import plot_funcs as p
f = Fitter(np.hstack((wls,wls)), t, np.hstack((pf, 2*sf)), True)
lm = f.start_lmfit([-0.05, 0.07, 2, 10, 6000],lower_bound=0.3, 
                   fixed_names=['t2']) 
lm.params['t1'].max=100
lm.params['t1'].min=5
lm.params['t0'].max=5

lm.leastsq()
figure(0)
p.plot_diagnostic(f)
figure(1)
plot(f.c[:,:800].T)
plot(f.c[:,800:].T, lw=2)
figure(2)
p.plot_das(f)