# -*- coding: utf-8 -*-
"""
This module helps to build a transfer matrix and applying it to an 
DAS.
"""
import sympy
import numpy as np
import scipy.linalg as la
from typing import List
import numbers

class Transition(object):
    """
    Represents a transtion between comparments.
    """
    def __init__(self, from_comp, to_comp, rate=None, qy=None):
        if rate is None:
            self.rate = sympy.Symbol(from_comp + '_' + to_comp, real=True, positive=True)
        else:
            self.rate = sympy.Symbol(rate, real=True, positive=True)
        self.from_comp = from_comp
        self.to_comp = to_comp
        self.qu_yield = qy or 1.


class Model(object):
    """
    Helper class to make a model
    """
    def __init__(self):
        self.transitions: List[Transition] = []

    def add_transition(self, from_comp, to_comp, rate=None, qy=None):
        """
        Adds an transition to the model.

        Parameters
        ----------
        from_comp : str
            Start of the transition
        to_comp : str
            Target of the transition
        rate : str, optional
            Name of the associated rate, by default None, which generates a
            default name.
        qy : str of float, optional
            The yield of the transition, by default 1
        """
        
        trans = Transition(from_comp, to_comp, rate, qy)
        self.transitions.append(trans)

    def build_matrix(self):
        """
        Builds the n x n k-matrix
        """
        comp = get_comparments(self.transitions)
        idx_dict = dict(enumerate(comp))
        inv_idx = dict(zip(idx_dict.values(), idx_dict.keys()))
        mat = sympy.zeros(len(comp))
        #for i, comp in idx_dict.iteritems():
        for t in self.transitions:
            i = inv_idx[t.from_comp]
            mat[i, i] = mat[i, i] - t.rate * t.qu_yield
            if t.to_comp != 'zero':
                mat[inv_idx[t.to_comp], i] += t.rate * t.qu_yield
                
        self.mat = mat
        return mat
    
    def build_mat_func(self):
        rates = set([t.rate for t in self.transitions])
        yields = (t.qu_yield for t in self.transitions if not isinstance(t.qu_yield, numbers.Number))
        print(rates, yields)
        params = list(rates) + list(yields)
        K = self.build_matrix()
        K_func = sympy.lambdify(params, K)
        return K_func

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
        (P, J) = (A * ts).jordan_form()
        out = sympy.zeros(P.cols)
        for i in range(P.cols):
            out[i, i] = sympy.exp(P[i, i])

        sim_Pinv = sympy.simplify(P.inv('ADJ'))
        sim_P = sympy.simplify(P)
        sol = (sim_P*out)*(sim_Pinv * y0)
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
        if trans.to_comp not in l and trans.to_comp != 'zero':
            l.append(trans.to_comp)
    return l


def get_symbols(list_trans):
    """
    Return the used symbols
    """
    return [t.rate for t in list_trans]

