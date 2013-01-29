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
    g.add_edge('S1_hot', 'S1_warm', tau=tau[3])
    g.add_edge('S1_hot', 'T_hot', tau=tau[0])
    g.add_edge('S1_warm', 'S1', tau=tau[1])
    g.add_edge('S1', 'S0', tau=6600)
    g.add_edge('T_hot', 'T1', tau=tau[2])
    
    #g.add_edge('S1', 'S0', tau=tau[2])
    a0 = nx.to_numpy_matrix(g, weight='tau')
    a = np.where(a0==0, 0, 1/a0)    
    a = a.T - np.diag(np.sum(a, 1).flat)
    return a, g
   

def sol_matexp(A, tlist, y0):
    w, v = np.linalg.eig(A)
    out = np.zeros((tlist.size, y0.size))
    viy0 = np.linalg.solve(v, y0)    
    for i, t in enumerate(tlist):
        sol_t = (np.dot(v, np.exp(-w*t))).dot(viy0)
        out[i, :] =  sol_t
    return out

def sol_matexp(A, tlist, y0):
    w, v = np.linalg.eig(A)
    iv = np.linalg.inv(v)
    out = np.zeros((tlist.size, y0.size))
    end = np.dot(np.linalg.inv(v), y0)
    for i, t in enumerate(tlist):
        sol_t = np.dot(v,np.diag(np.exp(w*t))).dot(iv).dot(y0)
        out[i, :] =  sol_t
    return out

from scipy.sparse.linalg import lsqr
import sklearn.linear_model as lm
import cvxpy 
sol = lm.ElasticNet(alpha=0.001, l1_ratio=1., 
                    positive=True, max_iter = 1e5, warm_start=1)
def fit(p, mode='lm'):
    a, g = make_A(p)       
    y0 = np.zeros(a.shape[0])
    print g.nodes()
    y0[g.nodes().index('S1_hot')] += 1       
    A = sol_matexp(a, f.t, y0)
    bleach = -A.sum(1)
    A = np.column_stack((A, bleach))
    N, M = A.shape[1], f.data.shape[1]
    c = np.empty((N, M))
    x = cvxpy.variable(N, 1)
    b = cvxpy.parameter(f.data.shape[0])             
    #b = cvxpy.matrix(f.data[:, 0:1])    
    A = cvxpy.matrix(A.copy())    
    lamb = cvxpy.ones(N) * 0.0001
    p = cvxpy.program(cvxpy.minimize(cvxpy.norm2(A*x-b+lamb*x)),
                          [cvxpy.geq(x, 0)])
    
    for i in range(M):
        b.value = cvxpy.matrix(f.data[:, i:i+1])            
        p.solve(quiet=True)   
        c[:, i] = x.value.squeeze()
    #c = ridge_regression(A, f.data, 0.001).T    
    res = f.data - A.dot(c)
    
    if mode=='sum':
        return (res*res).sum() #+ 15*np.abs(np.diff(c, 2, 1)).sum()
    elif mode=='p':
        return res, c, A, g
    elif mode=='lm':        
        o = res.flatten()
        o = np.array(o)[:]
        print o.shape
        #o[-1] += 0.01*c.sum()  
        return array(o[0,:])

import scipy.optimize as opt
x0 = [5, 10, 500, 0.3]
import lmfit
p = lmfit.Parameters()
for i, tau in enumerate(x0):
    p.add('tau'+str(i), tau, min=0)
x = opt.leastsq(fit, x0, 'lm')
#l = lmfit.Minimizer(fit, p, ['lm'])
#l.leastsq()
#x0 = [95., 11., 6600.]
#x = cma.fmin(fit, x0, 5, bounds=[0, None],restarts=1)
res, c, A, g = fit(x0, 'p')
clf()
subplot(131)
p1=plot(wl, c[:-1].T)
plot(wl, -c[-1].T)
xlabel('cm-1')
ylabel('OD')
subplot(132)
xlabel('t')
ylabel('conc')
plot(f.t, A)
xscale('log')
subplot(133)


for i in g.nodes():
    for j in g[i]:        
        g[i][j]['tau'] = '%2d'%g.edge[i][j]['tau']
        print g[i][j]['tau']

pos = {'S1_hot':(0, 3), 'S1_warm':(0,2.3),  'S1':(0, 1.5),
       'T_hot':(1, 1.5), 'T1':(1,1), 'S0': (0,0)}
#pos = nx.spring_layout(g, pos)
col = [i.get_color() for i in p1]
nx.draw(g, pos, node_size=3000, node_color=col)        
nx.draw_networkx_edge_labels(g, pos)
figure()
for i in [0, 5, 10, 20, -1]:
    plot(wl, f.data[i, :],'ro')
    plot(wl, (f.data - res)[i, :],'k')
    plot(wl, res[i,:])