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
        mat = sympy.zeros((len(comp), len(comp)))
        #for i, comp in idx_dict.iteritems():                     
        for t in self.transitions:
            i = inv_idx[t.from_comp]
            mat[i, i] = mat[i, i] - t.rate
            mat[inv_idx[t.to_comp], i] = t.rate 
            if t.qu_yield != 1. :
                mat[inv_idx[t.to_comp], i] = t.rate * t.qu_yield
        return mat
    
    def get_compartments(self):
        return get_comparments(self.transitions)
        
    def get_func(self, y0):
        """
        Gives back a function (compiled with cython)
        """        
        A = self.build_matrix()
        ts = sympy.Symbol('ts', real=True)   
        expA = (A*ts).exp()            
        fun = expA * y0          
        r, e = sympy.cse(fun)
        e_sim = [i.simplify() for i in e]
        prog_str = _make_progstr(self.transitions, fun, e, r)
        self.prog_str = prog_str
        self.fun = fun
        self.expA = expA
        #print prog_str
        def sol_fun(p, t):
            ret = cython.inline(prog_str)
            return ret
        return sol_fun #autowrap(fun,'C', backend='CYTHON', tempdir='..')

def _make_progstr(transitions, fun, e, r):    
    prog_str = 'out = np.zeros((t.size, {0}))\n'.format(fun.shape[0])
    #prog_str += 'exp = np.exp\n'
    prog_str += _make_appy_sym(get_symbols(transitions))
    prog_str += "for i in range(t.shape[0]):\n"
    prog_str += "   ts = t[i]\n"    
    for eq in r + list(zip(sympy.numbered_symbols("e"), e)):
            prog_str += '   %s = %s \n' % eq            
    for i in range(fun.shape[0]):
        prog_str += '   out[i, {0}] = e{0}'.format(str(i)) + '\n'
    prog_str += 'return out'
    print prog_str
    return prog_str


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
    model = Model()
    model.add_trans('a', 'b')
    model.add_trans('b', 'c')
    model.add_trans('c', 'd')
    model.add_trans('d', 'e')
    model.add_trans('e', 'e')
    #model.add_trans('S1', 'S0', tau=6600)
    t = np.linspace(1, 100, 1000)
    y0 = np.array([1, 0, 0, 0, 0])[:, None]
    #plot(t, model.exp_solution(y0, t))
    print get_comparments(model.transitions)
    print model.build_matrix()
    
    fu = model.get_func(y0)
    tau = np.array([1., 10., 100, 300, 10000, 10000])
    plot(t, fu(tau, t))