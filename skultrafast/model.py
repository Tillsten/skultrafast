# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:15:03 2013

@author: Tillsten
"""
import numpy as np
import networkx as nx
from sklearn.linear_model import ridge_regression
from skultrafast.kinetic_model import Model
import lmfit
from pylab import *


def sol_matexp(A, tlist, y0):
    w, v = np.linalg.eig(A)
    iv = np.linalg.inv(v)
    out = np.zeros((tlist.size, y0.size))    
    for i, t in enumerate(tlist):
        sol_t = np.dot(v,np.diag(np.exp(w*t))).dot(iv).dot(y0)
        out[i, :] =  sol_t
    return out


from skultrafast.lineshapes import lorentz

def make_state_spectra(para, x, state, shape='lorentz'):
    names = [state + '_' + i for i in ('A', 'w', 'xc')]
    vals = [para[i].value for i in names]    
    return lorentz(x, *vals)

def fit(p, x, mode='lm', has_spectral=True):
    taus = [p[name].value for name in p if name.startswith('tau')]   
    A = func(taus, f.t)    
    states = m.get_compartments()    
    if has_spectral:
        shapes = np.zeros((len(states), len(x)))
        for i, st in enumerate(states):        
            shapes[i, :] = make_state_spectra(p, x, st)
        bleach = make_state_spectra(p, x, 'bleach')
        bleach += p['const'].value
        #bleach2 = make_state_spectra(p, x, 'bleach2')
        #bleach += bleach2
        shapes[:-1,:] += bleach
        
        fit = A.dot(shapes)
        res = fit - f.data
        
    else:
        N, M = A.shape[1], f.data.shape[1]
        c = ridge_regression(A, f.data, 0.001).T    
        res = f.data - A.dot(c)
        
    if mode=='sum':
        return (res*res).sum() #+ 15*np.abs(np.diff(c, 2, 1)).sum()
    elif mode=='p':
        return res, A, shapes
    elif mode=='lm':        
        o = res.flatten()
        o = np.array(o)[:]        
        #o[-1] += 0.01*c.sum()  
        return o

#def make_A(tau):
#    g=nx.DiGraph()
#    g.add_edge('S1hot', 'S1warm', tau=tau[0])    
#    g.add_edge('S1hot', 'T1', tau=tau[1])
#    g.add_edge('S1warm', 'S1', tau=tauconda search --all[2])
#    g.add_edge('S1', 'S0', tau=7000)
#    #g.add_edge('S1', 'T2', tau=tau[2])
#    #g.add_edge('T2','T1', tau=tau[3])
#    #g.add_edge('S1', 'S0', tau=tau[2])
#    a0 = nx.to_numpy_matrix(g, weight='tau')
#    a = np.where(a0==0, 0, 1/a0)    
#    a = a.T - np.diag(np.sum(a, 1).flat)
#    return a, g
#   
m = Model()
m.add_trans('S1hot', 'S1warm')
#m.add_trans('S1hot', 'T1')
m.add_trans('S1warm', 'S1')
m.add_trans('S1', 'T1')
m.add_trans('T1', 'S0')
#m.add_trans('S0', 'S0')
print(m.get_compartments())
y0 = np.array([1., 0, 0, 0, 0])[:, None]
func = m.get_func(y0)
a = func(np.array([7., 8, 15, 6600]), f.t)
plot(f.t, a)
def add_state(p, state, A, w, xc, min=20, vary=True):
    p.add(state + '_A', A, min=min, vary=vary)
    p.add(state + '_w', w, min=2, vary=vary)
    p.add(state + '_xc', xc, min=0, vary=vary)


import scipy.optimize as opt
x0 = [5, 7, 15., 6600.]
import lmfit
p = lmfit.Parameters()

for i, tau in enumerate(x0):
    p.add('tau'+str(i), tau, min=0, vary=True)
p['tau3'].vary = False
add_state(p, 'S1hot', 30 , 4, 1516)
add_state(p, 'S1warm', 45, 4, 1519)
add_state(p, 'T1', 30, 4, 1515)
#add_state(p, 'T2', 40, 4, 1517)
add_state(p, 'S1', 45, 4, 1520)
add_state(p, 'S0', 0, 0, 0, vary=False)
add_state(p, 'bleach', -40, 4, 1522, min=None)
#add_state(p, 'bleach2', -3, 60, 1505, min=None)
p.add('const', -3)
mi = lmfit.Minimizer(fit, p, [f.wl])

mi.scalar_minimize('CG')
#
lmfit.report_errors(p, show_correl=False)
res, A, shapes = fit(p, f.wl, 'p')
print(sum(res**2))
subplot(121)
plot(f.wl, shapes.T)
subplot(122)
plot(f.t, A)
