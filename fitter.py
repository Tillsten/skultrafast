# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from numpy.core.umath_tests import matrix_multiply, inner1d
from scipy import linalg
#from scipy.special import erfc #errorfunction

from scipy.stats import f #fisher
from skultrafast import dv, zero_finding
import lmfit

LinAlgError = np.linalg.LinAlgError


#from skultrafast.base_functions import _fold_exp, _coh_gaussian
from skultrafast.base_functions_cl import _fold_exp, _coh_gaussian, _fold_exp_and_coh

posv = linalg.get_lapack_funcs(('posv'))

def direct_solve(a, b):
    c, x, info = posv(a, b, lower=False,
                      overwrite_a=True,
                      overwrite_b=False)
    return x

def solve_mat(A, b_mat, method='fast'):
    """
    Returns the solution for the least squares problem |Ax - b_i|^2.
    """
    if method == 'fast':
        #return linalg.solve(A.T.dot(A), A.T.dot(b_mat), sym_pos=True)
        return direct_solve(A.T.dot(A), A.T.dot(b_mat))

    elif method == 'ridge':
        alpha = 0.001
        X = np.dot(A.T, A)
        X.flat[::A.shape[1] + 1] += alpha
        Xy = np.dot(A.T, b_mat)
        #return linalg.solve(X, Xy, sym_pos=True, overwrite_a=True)
        return direct_solve(X, Xy)

    elif method == 'qr':
        cq, r = linalg.qr_multiply(A, b_mat)
        return linalg.solve_triangular(r, cq)

    elif method == 'cho':
        c, l = linalg.cho_factor( A.T.dot(A))
        return linalg.cho_solve((c, l), A.T.dot(b_mat))

    elif method == 'lstsq':
        return np.linalg.lstsq(A, b_mat)[0]

    elif method == 'lasso':
        import sklearn.linear_model as lm
        s = lm.Lasso(fit_intercept=False)
        s.alpha = 0.004
        s.fit(A, b_mat)
        return s.coef_.T

    else:
        raise ValueError('Unknow lsq method, use ridge, qr, fast or lasso')





class Fitter(object):
    """ The fit object, takes all the need data and allows to fit it.

    There a two different methods to fit the data. The fast one
    assumes, that the data has no dispersion, so the base vectors
    are the same for each channel. It is recommended to first work
    with the fast version. Note that the fast version is able to handle
    dispersion by using linear interpolation to transform the data
    to dispersion free data.

    The slower version calculates the base vector for each channel,
    in which the dispersion is integrated.

    The slower methods using the prefix full.

    Parameters
    ----------
    wl : ndarray(M)
        Array containing the wavelength-coordinates.
    t :  ndarray(N)
        Array containing the time-coordinates.
    data :  ndarry(N,M)
        The 2d-data to fit.
    model_coh : boolhttps://gist.github.com/Tillsten/00becc85f76c6d9ea2dd
        If the  model contains coherent artifacts at the time zero,
        defaults to False.
    model_disp : int
        Degree of the polynomial which models the dispersion. If 1,
        only a offset is modeled, which is very fast.
    """

    def __init__(self, tup, model_coh=False,  model_disp=1):

        wl, t, data = tup
        self.t = t
        self.wl = wl
        self.data = data
        self.verbose = False
        self.model_coh = model_coh
        self.model_disp = model_disp
        self.lsq_method = 'cho'

        self.num_exponentials = -1
        self.weights = None

        if model_disp > 1:
            self.org = data[:]
            self.disp_x = (wl - np.min(wl)) / (wl.max() - wl.min())
            self.used_disp = np.zeros(model_disp)


    def make_model(self, para):
        """
        Calculates the model for given parameters. After calling, the
        DAS is at self.c, the model at self.model.

        If the dispersion is
        modeled, it is done via linear interpolation. This way, the base-
        vectors and their decomposition are only calculated once.

        Parameters
        ----------
        para : ndarray(N)
            para has the following form:
                [p_0, ..., p_M, w, tau_1, ..., tau_N]
            Where p are the coefficients of the dispersion polynomial,
            w is the width of the system response and tau are the decay
            times. M is equal to self.model_disp.

        """
        self.last_para = np.asarray(para)
        if self._chk_for_disp_change(para):
            # Only calculate interpolated data if necessary:
            self.tn = np.poly1d(para[:self.model_disp])(self.disp_x)
            tup = dv.tup(self.wl, self.t, self.org)
            self.data = zero_finding.interpol(tup, self.tn)[2]
            self.used_disp[:] = para[:self.model_disp]

        self.num_exponentials = self.last_para.size - self.model_disp - 1
        if self.model_disp <= 1:
            self._build_xvec(para)
        self.x_vec = np.nan_to_num(self.x_vec)
        self.c = solve_mat(self.x_vec, self.data, self.lsq_method)
        self.model = np.dot(self.x_vec, self.c)
        self.c = self.c.T

    def _chk_for_disp_change(self, para):
        if self.model_disp > 1:
            if  np.any(para[:self.model_disp] != self.used_disp):
                return True
        return False


    def _build_xvec(self, para):
        """
        Build the base (the folded functions) for given parameters.
        """
        para = np.array(para)
        if self.verbose:
            print para

        try:
            idx = (para != self._last)
        except AttributeError:
            #self._l
            idx = [True] * len(para)

        if self.model_disp == 1:
            x0, w, taus = para[0], para[1], para[2:]
            tau_idx = idx[2:]
        else:
            x0, w, taus = 0., para[0], para[1:]
            tau_idx = idx[1:]

        if any(idx[:2]) or self.model_disp or True:
            if self.model_coh:
                x_vec = np.zeros((self.t.size, self.num_exponentials + 4))
                x_vec[:, -4:] = _coh_gaussian(self.t[:, None], w, x0).squeeze()
                x_vec[:, :-4] = _fold_exp(self.t[:, None], w,
                                               x0, taus).squeeze()
            else:
                x_vec = _fold_exp(self.t[:, None], w, x0, taus).squeeze()
            self.x_vec = np.nan_to_num(x_vec)
            self.x_vec /= np.max(self.x_vec, 0)
            self._last = para.copy()
        else:
            self.x_vec[:, tau_idx] = _fold_exp(self.t, w,
                                               x0, taus[tau_idx]).T

    def res(self, para):
        """
        Return the residuals for given parameters using the same
        basevector for each channel. See make_model for para format.
        """
        self.make_model(para)
        self.residuals = (self.model - self.data)
        if not self.weights is None:
            self.residuals *= self.weights
        return self.residuals.ravel()

    def full_res(self, para):
        """
        Return the residuals for given parameter modelling each
        channel for it own.
        """
        self.make_full_model(para)
        self.residuals = (self.model - self.data)
        if not self.weights is None:
            self.residuals *= self.weights
        return self.residuals.ravel()

    def make_full_model(self, para):
        """
        Calculates the model for given parameters. After calling, the
        DAS is at self.c, the model at self.model.

        Parameters
        ----------
        para : ndarray(N)
            para has the following form:
                [p_0, ..., p_M, w, tau_1, ..., tau_N]
            Where p are the coefficients of the dispersion polynomial,
            w is the width of the system response and tau are the decay
            times. M is equal to self.model_disp.

        """

        para = np.asarray(para)
        self._check_num_expontials(para)
        try:
            m_disp = self.model_disp
            is_disp_changed = (para[:m_disp] != self.last_para[:m_disp]).any()
        except AttributeError:
            is_disp_changed = True

        self.last_para = para
        print para
        if self.model_disp and is_disp_changed:
            self.tn = np.poly1d(para[:self.model_disp])(self.disp_x)
            self.t_mat = self.t[:, None] - self.tn[None, :]

        self._build_xmat(para[self.model_disp:], is_disp_changed)

        for i in xrange(self.data.shape[1]):
            A = self.xmat[:, i, :]
            self.c[i, :] = solve_mat(A, self.data[:, i], self.lsq_method)

        #self.model[:, i] = self.xmat[:, i, :].dot(self.c[i, :])

        self.model = inner1d( self.xmat, self.c)
        #self.model[:, :]  = matrix_multiply(self.xmat, self.c[:, :, None]).squueze()

    def _build_xmat(self, para, is_disp_changed):
        """
        Builds the basevector for every channel. The vectors
        are save self.xmat.
        """
        para = np.array(para)
        try:
            idx = (para != self._last)
        except AttributeError:
            idx = [True] * len(para)

        w = para[0]
        taus = para[1:]
        x0 = 0.

        #Only calculate what is necessary.
        if idx[0] or is_disp_changed:
            exps, coh = _fold_exp_and_coh(self.t_mat, w, x0, taus)
            if self.model_coh:
                self.xmat[:, :, -4:] = coh
            num_exp = self.num_exponentials
            self.xmat[:, :, :num_exp] =  exps
        elif any(idx):
            self.xmat[:, :, idx[1:]] = _fold_exp(self.t_mat, w,
                                                 x0, taus[idx[1:]])
        #self.xmat = np.nan_to_num(self.xmat)
        self._last = para

    def _check_num_expontials(self, para):
        """
        Check if num_exp changed and allocate space as necessary.
        """
        new_num_exp = para.size - self.model_disp - 1
        if new_num_exp != self.num_exponentials:
            self.num_exponentials = new_num_exp
            if self.model_disp:
                new_num_exp += 4
            n, m = self.data.shape
            self.xmat = np.empty((n, m, new_num_exp))
            self.c = np.zeros((self.data.shape[1], self.xmat.shape[-1]))
            self.model = np.empty_like(self.data)

    def res_sum(self, para):
        """Returns the squared sum of the residuals for given parameters"""
        return np.sum(self.res(para) ** 2)

    def start_lmfit(self, x0, fixed_names=[], lower_bound=0.3,
                    fix_long=True, fix_disp=False, full_model=1):
        p = lmfit.Parameters()
        for i in range(self.model_disp):
            p.add('p' + str(i), x0[i])
            if fix_disp:
                p['p' + str(i)].vary = False
        x0 = x0[self.model_disp:]

        p.add('w', x0[0], min=0)
        num_exp = len(x0) - 1
        for i, tau in enumerate(x0[1:]):
            name = 't' + str(i)#
#
            p.add(name, tau, vary=True)
            if name not in fixed_names:
                p[name].min = lower_bound
            else:
                p[name].vary = False

        for i in fixed_names:
            p[i].vary = False
        if fix_long:
            p['t' + str(num_exp - 1)].vary = False

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

def start_pymc(fitter, x0, bounds):
    import pymc

    rs = [(pymc.Uniform('r' + str(i), lower, upper)) for
          (i, (lower, upper)) in enumerate(bounds)]
    z0 = pymc.Uniform('z0', -1, 1)
    sig = pymc.Uniform('sig', 0, 0.15)
    tau = pymc.Uniform('tau', 0, 25, size=fitter.data.shape[1])
    #tau.value=20*self.data.shape[1]
    H = lambda x: fitter.model(x)

    @pymc.deterministic
    def mod(z0=z0, sig=sig, rs=rs):
        x = np.array([z0, sig] + rs)
        H(x)
        return fitter.model

    l = []
    for i in range(fitter.data.shape[1]):
        l.append(pymc.Normal(observed=True, name='res' + str(i),
                             value=fitter.data[:, i], tau=tau[i],
                             mu=mod[:, i]))

    mo = pymc.Model(set([z0, tau] + rs + l))
    return mo



def f_compare(Ndata, Nparas, new_chi, best_chi, Nfix=1.):
    """
    Returns the probalitiy for two given parameter sets.
    Nfix is the number of fixed parameters.
    """
    Nparas = Nparas + Nfix
    return f.cdf((new_chi / best_chi - 1) * (Ndata - Nparas) / Nfix,
                 Nfix, Ndata - Nparas)



def mod_dof_f(dof):
    def f_mod(N, P, chi, old_chi, Nfix=1.):
        return f_compare(N, P + dof, chi, old_chi, Nfix)
    return f_mod


if __name__ == '__main__':
    pass
    #import pymc
#
#    coef = np.zeros((2, 400))
#    coef[0, :] = -np.arange(-300, 100) ** 2 / 100.
#    coef[1, :] = np.arange(-200, 200) ** 2 / 100.
#    t = np.linspace(0, 30, 300)
#    g = Fitter(np.arange(400), t, 0, False)
#    g.build_xvec([0.1, 0.3, 5, 16])
#    dat = np.dot(g.x_vec, coef)
#
#    dat += 10 * (np.random.random(dat.shape) - 0.5)
#    dat = dat * (1 + (np.random.random(dat.shape) - 0.5) * 0.20)
#    g = Fitter(np.arange(400), t, dat, 2, False, False)
#    x0 = [0.5, 0.2, 4, 20]

    #a = g.start_pymc(x0, [(0.2, 20), (0.2, 20)])
    #b = pymc.MCMC(a)
    #b.isample(10000, 1000)
    #pymc.Matplot.plot(b)
    #    #a=g.start_cmafit(x0)
#    a = g.start_lmfit(x0)
 #   a.leastsq()
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
