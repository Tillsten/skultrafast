# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from numpy.linalg import solve, qr
from scipy.special import erfc #errorfunction
from scipy.optimize import leastsq
from scipy.stats import f #fisher
#from scipy.linalg import qr, lstsq
LinAlgError = np.linalg.LinAlgError
#import scipy.sparse.linalg as li
from sklearn.linear_model.ridge import ridge_regression
import dv, zero_finding
import lmfit
import cython
sq2 = np.sqrt(2)
BASE_WL = 550.
import numexpr as ne
ne.set_vml_accuracy_mode('fast')
def _fold_exp(tt, w, tz, tau):
    """
    Returns the values of the folded exponentials for given parameters.

    Parameters
    ----------
    tt:  ndarray(N)
        Array containing the time-coordinates
    w:  float
        The assumed width/sq2
    tz: float
        The assumed time zero.
    tau: ndarray(M)
        The M-decay rates.

    Returns
    -------
    y: ndarray
       Folded exponentials for given taus.
"""

    ws = w
    k = 1 / (tau[..., None, None])
    t = (tt + tz).T[None, ...]
    y = ne.evaluate('exp(k * (ws * ws * k / (4.0) - t))')
    y *= 0.5 * erfc(-t / ws + ws * k / (2.0))#/(ws*np.sqrt(2*np.pi))
    #y /= np.max(np.abs(y), 0)
    return y.T


def _exp(tt, w, tz, tau):
    """
    Returns the values of the exponentials for given parameters.

    Parameters
    ----------
    tt:  ndarray(N)
        Array containing the time-coordinates
    w:  float
        The assumed width/sq2
    tz: float
        The assumed time zero.
    tau: ndarray(M)
        The M-decay rates.

    Returns
    -------
    y: ndarray
       Exponentials for given tau's.
    """
    t = tt + tz
    y = np.exp(-t / (tau[:, None]))
    return y


def _coh_gaussian(t, w, tz):
    """Models artifacts proportional to a gaussian and it's derivatives

    Parameters
    ----------
    t:  ndarray
        Array containing the time-coordinates
    w:  float
        The assumed width/sq2
    tz: float
        The assumed time zero.

    Returns
    -------
    y:  ndarray (len(t), 4)
        Array containing a gaussian and it the scaled derivatives,
        each in its own column.

    """
    w = w / sq2
    tt = t + tz
    y = ne.evaluate('where(tt/w < 4, exp(-0.5 * (tt / w) ** 2) / (w * sqrt(2 * 3.14159265)), 0)')    
    y = np.tile(y[..., None], (1, 1, 4))
    y[..., 1] *= (-tt / w ** 2)
    y[..., 2] *= (tt ** 2 / w ** 4 - 1 / w ** 2)
    y[..., 3] *= (-tt ** 3 / w ** 6 + 3 * tt / w ** 4)
    y /= np.max(np.abs(y), 0)
    return y


class Fitter(object):
    """ The fit object, takes all the need data and allows to fit it.

    Parameters
    ----------
    wl : ndarray(M)
        Array containing the wavelength-coordinates.
    t :  ndarray(N)
        Array containing the time-coordinates.
    data :  ndarry(N,M)
        The 2d-data to fit.
    model_coh : bool
        If the  model contains coherent artifacts at the time zero, defaults to False.
    bounds : float
        Bounds to use for constraint fitting of the linear coefficients, is
        only used in con_model.
    """

    def __init__(self, wl, t, data,
                 model_coh=False, model_disp=0, bound=1000., tn=None):
        self.t = t
        self.wl = wl
        self.model_coh = model_coh
        self.model_disp = model_disp
        self.data = data

        #self.one=np.identity(t.size)

        self.xmat = np.empty((1))
        self.weights = None
        
        if model_disp:
            self.org = data[:]
            self.disp_x = (wl - np.min(wl)) / (wl.max() - wl.min())
            self.used_disp = np.zeros(model_disp)
        

    def make_model(self, para):
        """
        Returns the fit for given system parameters.

        para has the following form:
        [xc,w,tau_1,...tau_n]
        If model_disp is True than there are  model_disp-parameters are at the beginning.
        """
        self.last_para = para
        if self.model_disp:
            if  np.any(para[:self.model_disp] != self.used_disp):
                self.tn = np.poly1d(para[:self.model_disp])(self.disp_x)
                tup = dv.tup(self.wl, self.t, self.org)                
                self.data  = zero_finding.interpol(tup, self.tn)[2]                
                self.used_disp[:] = para[:self.model_disp]
            para = para[self.model_disp:]

        self.build_xvec(para)      
       
        q, r = np.linalg.qr(self.x_vec)
        self.c = np.linalg.solve(r, q.T.dot(self.data))
        #self.c = np.linalg.pinv(self.x_vec).dot(self.data)
        self.m = np.dot(self.x_vec, self.c)
        
    def full_res(self, para):        
        self.make_full_model(para)
        
        self.residuals = (self.model - self.data)
        if not self.weights is None:
            self.residuals /= self.weights
        return self.residuals.flatten()
    
    def make_full_model(self, para):  
        from sklearn.linear_model import Lasso
        para = np.asarray(para)
        mdisp = self.model_disp
        try:
            is_disp_changed = (para[:mdisp] != self.last_para[:mdisp]).any()
        except AttributeError:
            is_disp_changed = True
        self.last_para = para
        if self.model_disp:
            self.tn = np.poly1d(para[:self.model_disp])(self.disp_x)            
        self.t_mat = self.t[:, None] - self.tn[None, :]
        self.build_xmat(para[self.model_disp:], is_disp_changed)
        if not hasattr(self, 'c'):
            self.c = np.zeros((self.data.shape[1], self.xmat.shape[-1]))
            self.model = np.empty_like(self.data)
        for i in xrange(self.data.shape[1]):            
            #
            try:
                A = self.xmat[:, i, :]
                #l = Lasso(alpha=0.0001)
                #l.fit(A, self.data[:, i], coef_init=self.c[i, :])
                #self.c[i, :] = l.coef_
                self.c[i, :] = ridge_regression(A, self.data[:, i], 0.00001)                
                #self.c[i, :] = np.linalg.solve(A.T.dot(A), A.T.dot(self.data[:, i]))                    
            except LinAlgError:
                q, r = np.linalg.qr(self.xmat[:, i, :])                 
                self.c[i, :] = np.linalg.solve(r, q.T.dot(self.data[:, i]))  
            self.model[:, i] = self.xmat[:, i, :].dot(self.c[i, :])




    def _check_xmat(self):
        """
        Makes new xmat if  the number of exponentials changes.
        """
        n, m = self.t_mat.shape
        num_exp = self.num_exponentials
        if self.model_disp:
            num_exp += 4
        if self.xmat.shape != (n, m, num_exp) or self.xmat is None:
            self.xmat = np.empty((n, m, num_exp))

    def build_xmat(self, para, is_disp_changed):
        """
        Builds the basevectors for every channel.
        """        
        para = np.array(para)
        self.num_exponentials = para.size - 1       
        self._check_xmat()
        try: 
            idx = (para != self._last)            
        except AttributeError:
            idx = [True] * len(para)        
        
        w = para[0]                
        taus = para[1:]
        x0 = 0
        print w, taus
        if idx[0] or is_disp_changed:  
            if self.model_coh:
                self.xmat[:, :, -4:] = _coh_gaussian(self.t_mat, w, x0)
            self.xmat[:, :, :self.num_exponentials] = _fold_exp(self.t_mat, w, 
                                                                x0, taus)
        elif any(idx):                        
            self.xmat[:, :, idx[1:]] = _fold_exp(self.t_mat, w, x0, taus[idx[1:]])
        self.xmat = np.nan_to_num(self.xmat)
        self._last = para
        
    def build_xvec(self, para):
        """
        Build the base (the folded functions) for given parameters.
        """        
        para = np.array(para)
        try: 
            idx = (para != self._last)            
        except AttributeError:
            idx = [True] * len(para)
        if any(idx[:2]) or self.model_disp:
            self.num_exponentials = para.size - 2
            if self.model_coh:
                self.x_vec = np.zeros((self.t.size, self.num_exponentials + 4))
                self.x_vec[:, -4:] = _coh_gaussian(self.t, para[1], para[0])
                self.x_vec[:, :-4] = _fold_exp(self.t, para[1],
                                               para[0], (para[2:])).T
            else:
                self.x_vec = _fold_exp(self.t, para[1], para[0], (para[2:])).T
            self.x_vec = np.nan_to_num(self.x_vec)
            self._last = para.copy()
        else:           
            self.x_vec[:, idx[2:]] = _fold_exp(self.t, para[1],
                                            para[0], para[idx]).T
        
    def res(self, para):
        """Return the residuals for given parameters."""
        self.make_model(para)
        self.residuals = (self.data - self.m)
        return (self.residuals / self.weights).flatten()

    def res_sum(self, para):
        """Returns the squared sum of the residuals for given parameters"""
        return np.sum(self.res(para) ** 2)


    def start_lmfit(self, x0, fixed_names=[], lower_bound=0.3, 
                    fix_long=True, full_model=1):
        p = lmfit.Parameters()
        for i in range(self.model_disp):
            p.add('p' + str(i), x0[i])        
        x0 = x0[self.model_disp:]
        
        if not full_model:
            p.add('x0', x0[0])
            x0 = x0[1:]
        
        p.add('w', x0[0], min=0)
        num_exp = len(x0) - 1
        for i, tau in enumerate(x0[1:]):
            name = 't' + str(i)
            p.add(name, tau, vary=True)
            if tau not in fixed_names:
                p[name].min = lower_bound
                
        for k in fixed_names:
            p[k].vary = False
        
        if fix_long:
            p['t'+str(num_exp - 1)].vary = False
        
        def res(p):
            x = [k.value for k in p.values()]
            return self.res(x)
        
        def full_res(p):
            x = [k.value for k in p.values()]
            return self.full_res(x)
        fun = full_res if full_model else res
        
        return lmfit.Minimizer(fun, p)


    def start_cmafit(self, x0, restarts=2):
        import cma

        out = cma.fmin(self.res_sum, x0, 2, verb_log=0, verb_disp=50,
            restarts=restarts, tolfun=1e-6, tolfacupx=1e9)
        for pi in (out[0]):
            print "{0: .3f} +- {1:.4f}".format(pi, np.exp(pi))

        return out


    def start_pymc(self, x0, bounds):
        import pymc

        rs = [(pymc.Uniform('r' + str(i), lower, upper)) for (i, (lower, upper)) in enumerate(bounds)]
        z0 = pymc.Uniform('z0', -1, 1)
        sig = pymc.Uniform('sig', 0, 0.15)
        tau = pymc.Uniform('tau', 0, 25, size=self.data.shape[1])
        #tau.value=20*self.data.shape[1]
        H = lambda x: self.model(x)

        @pymc.deterministic
        def mod(z0=z0, sig=sig, rs=rs):
            x = np.array([z0, sig] + rs)
            H(x)
            return self.m.T

        l = []
        for i in range(self.data.shape[1]):
            l.append(pymc.Normal(observed=True, name='res' + str(i),
                value=self.data[:, i], tau=tau[i], mu=mod[:, i]))

        mo = pymc.Model(set([z0, tau] + rs + l))

        return mo
        
        

class ExactFitter(Fitter):
    pass

    


def f_compare(Ndata, Nparas, new_chi, best_chi, Nfix=1.):
    """
    Returns the probalitiy for two given parameter sets.
    Nfix is the number of fixed parameters.
    """

    # print new_chi, best_chi, Ndata, Nparas

    Nparas = Nparas + Nfix
    return f.cdf((new_chi / best_chi - 1) * (Ndata - Nparas) / Nfix,
        Nfix, Ndata - Nparas)


def mod_dof_f(dof):
    def f_mod(N, P, chi, old_chi, Nfix=1.):
        return f_compare(N, P + dof, chi, old_chi, Nfix)
    return f_mod


if __name__ == '__main__':
    #import pymc

    coef = np.zeros((2, 400))
    coef[0, :] = -np.arange(-300, 100) ** 2 / 100.
    coef[1, :] = np.arange(-200, 200) ** 2 / 100.
    t = np.linspace(0, 30, 300)
    g = Fitter(np.arange(400), t, 0, False)
    g.build_xvec([0.1, 0.3, 5, 16])
    dat = np.dot(g.x_vec, coef)

    dat += 10 * (np.random.random(dat.shape) - 0.5)
    dat = dat * (1 + (np.random.random(dat.shape) - 0.5) * 0.20)
    g = Fitter(np.arange(400), t, dat, 2, False, False)
    x0 = [0.5, 0.2, 4, 20]
    
    #a = g.start_pymc(x0, [(0.2, 20), (0.2, 20)])
    #b = pymc.MCMC(a)
    #b.isample(10000, 1000)
    #pymc.Matplot.plot(b)
#    #a=g.start_cmafit(x0)
    a=g.start_lmfit(x0)
    a.leastsq()
#    lmfit.printfuncs.report_errors(a.params)
#    #ar=g.chi_search(a[0])
#    import matplotlib.pyplot as plt
#
##    def plotxy(a):
##        plt.plot(a[:,0],a[:,1])
##    #
##    for i in range(len(a[0])-1):
##        plt.subplot(2,2,i+1)
##        plotxy(ar[i])
#plt.tight_layout()
#plot_das(g,1)
#plot_diagnostic(g)
#plot_spectra(g)
#wls=[30,70,100]
#plot_transients(g,wls)
#plt.show()
#best=leastsq(g.varpro,x0, full_output=True)
