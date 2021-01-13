# %%
from lmfit.parameter import Parameters
from numpy.lib.function_base import trim_zeros
from skultrafast import plot_helpers

import torch, attr, math
import numpy as np
from skultrafast.dataset import TimeResSpec
from scipy.optimize import least_squares
import lmfit
from typing import Optional, Callable
exp_half = math.exp(1 / 2.)

@torch.jit.script
def lstsq(b, y, alpha: float=0.1):
    """
    Batched linear least-squares for pytorch with optional L1 regularization.

    Parameters
    ----------

    b : shape(L, M, N) 
    y : shape(L, M)

    Returns
    -------
    tuple of (coefficients, model, residuals)

    """
    bT = b.transpose(-1, -2)
    AA = torch.bmm(bT, b)

    if alpha != 0:
        diag = torch.diagonal(AA, dim1=1, dim2=2)
        diag += alpha
    RHS = torch.bmm(bT, y[:, :, None])
    X, LU = torch.solve(RHS, AA)
    fit = torch.bmm(b, X)[..., 0]
    res = y - fit
    return X[..., 0], fit, res

@torch.jit.script
def make_base(tt, w, tau, model_coh: bool=False) -> torch.Tensor:
    """
    Calculates the basis for the linear least squares problem

    Parameters
    ----------
    tt : ndarry 
        2D-time points
    w : float
        System response
    tau : ndarray
        1D decay times
    model_coh : bool        
        If true, appends a gaussian and its first two 
        dervitates to model coherent behavior at time-zero.


    Returns
    -------
    [type]
        [description]
    """

    k = 1 / (tau[None, None, ...])
    
    t = (tt)[..., None]
    scaled_tt = tt / w
    if False:
        A = torch.exp(-k * tt)
    else:
        nw = w[:, None]
        A = 0.5 * torch.erfc(-t / nw + nw * k / (2.0))
        A *= torch.exp(k * (nw * nw * k / (4.0) - t)) 
        
    if model_coh:
        exp_half = torch.exp(0.5)
        scaled_tt = tt / w
        coh = torch.exp(-0.5 * scaled_tt * scaled_tt)
        coh = coh[:, :, None].repeat((1, 1, 3))
        coh[..., 1] *= (-scaled_tt * exp_half)
        coh[..., 2] *= (scaled_tt - 1)
        A = torch.cat((A, coh), dim=-1)

    #if torch.isnan(A).any():
    #    print(A)
    torch.nan_to_num(A, out=A)
    return A


@attr.s(auto_attribs=True)
class FitterTorch:
    dataset: TimeResSpec = attr.ib()
    buf_dataset: Optional[TimeResSpec] = None
    zero_func: Callable = attr.ib(lambda x: np.zeros_like(x))
    done_eval: bool = attr.ib(False)
    use_cuda: Optional[bool] = attr.ib(None)
    disp_poly_deg: int = attr.ib(2)
    sigma_deg: int = attr.ib(1)
    model_coh: bool = attr.ib(False)
    extra_base: Optional[np.ndarray] = attr.ib(None)

    def __attrs_post_init__(self):
        ds = self.dataset
        self.dev_data = torch.from_numpy(ds.data.T)
        if self.extra_base is not None:
            self.extra_base = torch.from_numpy(self.extra_base)
            
        if self.use_cuda:
            self.dev_data = self.dev_data.cuda()
            if self.extra_base is not None:
                self.extra_base.cuda()

    def eval(self, tt, w, tau, buffer=False):
        """
        Evaluates a model for given arrays

        Parameters
        ----------
        tt : ndarray
            Contains the delay-times, should have the same shape as the data.
        w : float
            The IRF width.
        tau : ndarray
            Contains the decay times.
        """

        tt = torch.from_numpy(tt)
        tau = torch.from_numpy(tau)
        w = torch.from_numpy(w)[:, None]
        if self.use_cuda:
            tt = tt.cuda()
            tau = tau.cuda()
            w = w.cuda()
            tau = tau.cuda()    
            
        A = make_base(tt, w, tau, self.model_coh)
        if self.extra_base is not None:
            if self.use_cuda:
                self.extra_base = self.extra_base.cuda()
            
            A = torch.cat((A, self.extra_base), dim=2)
        X, fit, res = lstsq(A, self.dev_data)
        self.A = A
        self.done_eval = True
        self.c = X
        self.model = fit
        self.residuals = res
        return X, fit, res

    def fit_func(self, x):
        ds = self.dataset
        self.disp_coefs = x[:self.disp_poly_deg]        
        taus = x[self.disp_poly_deg:-self.sigma_deg]
        w_coefs = x[-self.sigma_deg:]
        x = ds.wavenumbers
        xn = 2*(x - x.min()) / x.ptp()-1
        self.t_zeros = np.poly1d(self.disp_coefs)(xn)
        if len(w_coefs) > 1:
            self.w = np.poly1d(w_coefs)(xn)            
            self.w = np.maximum(self.w, 0.01)
        else:
            self.w = w_coefs
        self.tt = np.subtract.outer(ds.t, self.t_zeros).T        
        c, model, res = self.eval(self.tt, self.w, taus, True)
        return res.cpu().numpy().ravel()


    def start_lmfit(self,
                    w,
                    taus,
                    fix_last_tau=False,
                    fix_width=False,
                    fix_disp=False,
                    disp_params=None,
                    least_squares_kw=None):

        ds = self.dataset
        if disp_params is None:
            time_zeros = self.zero_func(ds.wavenumbers)
            x = ds.wavenumbers
            xn = (x - x.min()) / x.ptp()
            disp_guess = np.polyfit(xn, time_zeros, self.disp_poly_deg - 1)
        else:
            disp_guess = disp_params

        paras = lmfit.Parameters()
        for i, p in enumerate(disp_guess):
            paras.add('p%d' % i, value=p, vary=not fix_disp)        
        for i, p in enumerate(taus):
            fixed = fix_last_tau & (i == len(taus) - 1)
            paras.add('tau_%d' % i, min=0.01, value=p, vary=not fixed)
        self.sigma_deg = len(w)
        for i, p in enumerate(w):
            paras.add('w%d' % i, value=p, vary=not fix_disp)

        def fix_func(x: lmfit.Parameters):
            x = np.array(x)
            return self.fit_func(x)

        mini = lmfit.Minimizer(fix_func, paras)
        return paras, mini


