# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:15:03 2013

@author: Tillsten
"""
import numpy as np
import networkx as nx
from sklearn.linear_model import ridge_regression

def make_A(tau):
    g=nx.DiGraph()
    g.add_edge('S1_hot', 'S1_warm', tau=tau[0])    
    g.add_edge('S1_warm', 'S1', tau=tau[1])
    g.add_edge('S1', 'S0', tau=6000)
    g.add_edge('S1', 'T2', tau=tau[2])
    #g.add_edge('T2','T1', tau=tau[3])
    #g.add_edge('S1', 'S0', tau=tau[2])
    a0 = nx.to_numpy_matrix(g, weight='tau')
    a = np.where(a0==0, 0, 1/a0)    
    a = a.T - np.diag(np.sum(a, 1).flat)
    return a, g
   


def sol_matexp(A, tlist, y0):
    w, v = np.linalg.eig(A)
    iv = np.linalg.inv(v)
    out = np.zeros((tlist.size, y0.size))    
    for i, t in enumerate(tlist):
        sol_t = np.dot(v,np.diag(np.exp(w*t))).dot(iv).dot(y0)
        out[i, :] =  sol_t
    return out


from skultrafast.lineshape import lorentz

def make_state_spectra(para, x, state, shape='lorentz'):
    names = [state + '_' + i for i in ('A', 'w', 'xc')]
    vals = [para[i].value for i in names]    
    return lorentz(x, *vals)

def fit(p, x, mode='lm', has_spectral=True):
    taus = [p[name].value for name in p if name.startswith('tau')]
    a, g = make_A(taus)       
    y0 = np.zeros(a.shape[0])
    states =  g.nodes()
    y0[states.index('S1_hot')] += 1       
    A = sol_matexp(a, f.t, y0)
    s0 = states.index('S0')
    if has_spectral:
        shapes = np.zeros((len(states), len(x)))
        for i, st in enumerate(states):        
            shapes[i, :] = make_state_spectra(p, x, st)
        bleach = make_state_spectra(p, x, 'bleach')
        bleach2 = make_state_spectra(p, x, 'bleach2')
        bleach += bleach2
        shapes = shapes + bleach
        shapes[s0, :] -= bleach
        fit = A.dot(shapes)
        res = fit - f.data
        
    else:
        N, M = A.shape[1], f.data.shape[1]
        c = ridge_regression(A, f.data, 0.001).T    
        res = f.data - A.dot(c)
        
    if mode=='sum':
        return (res*res).sum() #+ 15*np.abs(np.diff(c, 2, 1)).sum()
    elif mode=='p':
        return res, A, g, shapes
    elif mode=='lm':        
        o = res.flatten()
        o = np.array(o)[:]
        print o.shape
        #o[-1] += 0.01*c.sum()  
        return o

def add_state(p, state, A, w, xc, min=0, vary=True):
    p.add(state + '_A', A, min=min, vary=vary)
    p.add(state + '_w', w, min=0, vary=vary)
    p.add(state + '_xc', xc, min=0, vary=vary)

import scipy.optimize as opt
x0 = [2, 5, 100, 5]
import lmfit
p = lmfit.Parameters()

for i, tau in enumerate(x0):
    p.add('tau'+str(i), tau, min=0)
add_state(p, 'S1_hot', 40, 4, 1516)
add_state(p, 'S1_warm', 45, 4, 1519)
add_state(p, 'T1', 50, 4, 1515)
add_state(p, 'T2', 40, 4, 1515)
add_state(p, 'S1', 45, 4, 1520)
add_state(p, 'S0', 0, 0, 0, vary=False)
add_state(p, 'bleach', -40, 6, 1522, min=None)
add_state(p, 'bleach2', -0, 6, 1505, min=None)

mi = lmfit.Minimizer(fit, p, [f.wl])

mi.leastsq()
lmfit.report_errors(p)


res, A, g, shapes = fit(p, f.wl, 'p')
