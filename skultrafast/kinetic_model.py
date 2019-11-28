# -*- coding: utf-8 -*-
"""
This modul build a kinetic matrix from given transtions between
compartments.
"""
import sympy, cython
import numpy as np


class Transition(object):
    """
    Represents a transtion between comparments.
    """
    def __init__(self, from_comp, to_comp,
                 tau=None, rate=None, qu_yield=None):
        self.rate = sympy.Symbol(from_comp + '_' + to_comp, real=True,
                                 positive=True)
        self.from_comp = from_comp
        self.to_comp = to_comp
        self.qu_yield = qu_yield or 1.


class Model(object):
    """
    Helper class to make a model
    """
    def __init__(self):
        self.transitions = []

    def add_trans(self, *args, **kwargs):
        """
        Creates a transiton and adds it to self
        """
        trans = Transition(*args, **kwargs)
        self.transitions.append(trans)

    def build_matrix(self):
        """model.add_trans('d', 'e')
        Builds the n x n k-matrix
        """
        comp = get_comparments(self.transitions)
        idx_dict = dict(enumerate(comp))
        inv_idx = dict(zip(idx_dict.values(), idx_dict.keys()))
        mat = sympy.zeros(len(comp))
        #for i, comp in idx_dict.iteritems():
        for t in self.transitions:
            i = inv_idx[t.from_comp]
            mat[i, i] = mat[i, i] - t.rate
            mat[inv_idx[t.to_comp], i] = t.rate
            if t.qu_yield != 1. :
                mat[inv_idx[t.to_comp], i] = t.rate * t.qu_yield
        self.mat = mat
        return mat

    def get_compartments(self):
        return get_comparments(self.transitions)

    def make_diff_equation(self):
        A = self.build_matrix()
        funcs = []
        t = sympy.Symbol('t', real=True)
        for c in self.get_compartments():
            funcs.append(sympy.Function(c)(t))
        eqs = []
        for i, row in enumerate(A):

            eqs.append(sympy.Eq(sympy.diff(funcs[i]), row.sum()))
        print(eqs)




    def get_func(self, y0=None):
        """
        Gives back a function (compiled with cython)
        """

        A = self.build_matrix()
        if y0 is None:
            y0 = sympy.zeros(A.shape[0])
            y0[0] = 1


        ts = sympy.Symbol('ts', real=True, positive=True)
        (P, J ) = (A*ts).jordan_form()
        out = sympy.zeros(P.cols)
        for i in range(P.cols):
            out[i,i] = sympy.exp(P[i,i])
        print(out, '\n', sympy.simplify(out))
        sim_Pinv = sympy.simplify(P.inv('ADJ'))
        sim_P = sympy.simplify(P)
        sol = (sim_P*out*sim_Pinv)*y0
        #print(sol)
        print(sympy.cse(sol))
        return sol




    def get_trans(self, y0, taus, t):
        """
        Return the solution
        """
        symbols = get_symbols(self.transitions)
        k = np.array(self.mat.subs(zip(symbols, taus))).astype('float')
        o = np.zeros((len(t), k.shape[0]))

        for i in range(t.shape[0]):
            o[i, :] = la.expm(k * t[i]).dot(y0)[:, 0]

        return o
       #pass  print mat.

import scipy.linalg as la


def _make_appy_sym(sym):
    l = []
    for i, s in enumerate(sym):
        l.append(str(s) + '= 1/p[{}]\n'.format(i))
    return ''.join(l)

def get_comparments(list_trans):
    """
    Getting a list of transtions, return the compartments
    """
    l = []
    for trans in list_trans:
        if trans.from_comp not in l:
            l.append(trans.from_comp)
        if trans.to_comp not in l:
            l.append(trans.to_comp)
    return l

def get_symbols(list_trans):
    """
    Return the used symbols
    """
    return [t.rate for t in list_trans]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    model = Model()
    model.add_trans('a', 'b')
    model.add_trans('b', 'c')
    model.add_trans('c', 'd')
    model.add_trans('d', 'e')
    #model.add_trans('e', 'e')
    #model.add_trans('S1', 'S0', tau=6600)
    t = np.linspace(1, 100, 1000)
    #model.make_diff_equation()
    #plt.plot(t, model.exp_solution(y0, t))
    print(get_comparments(model.transitions))
    print(model.build_matrix())

    fu = model.get_func()
#    tau = np.array([1., 10., 100, 300, 10000, 10000])
#    model.get_trans(y0, tau, t)