import torch, attr, math
import numpy as np
from skultrafast.dataset import TimeResSpec
from scipy.optimize import least_squares

from typing import Optional, Callable, Enu
exp_half = math.exp(1 / 2.)


def lstsq(b, y, alpha=0.01):
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
    X, LU = torch.gesv(RHS, AA)
    fit = torch.bmm(b, X)[..., 0]
    res = y - fit
    return X[..., 0], fit, res


@attr.s(auto_attribs=True)
class FitterTorch:
    dataset: TimeResSpec = attr.ib()
    zero_func: callable = attr.ib(lambda x: np.zeros_like(x))
    done_eval: bool = attr.ib(False)
    use_cuda: Optional[bool] = attr.ib(None)
    disp_poly_deg: int = attr.ib(2)
    model_coh: bool = attr.ib(0)

    def __attrs_post_init__(self):
        ds = self.dataset
        self.dev_data = torch.from_numpy(ds.data.T)
        if self.use_cuda:
            self.dev_data = self.dev_data.cuda()

    def eval(self, tt, w, tau, model_coh=False):
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
        if self.use_cuda:
            tt = tt.cuda()
            tau = tau.cuda()

        k = 1 / (tau[None, None, ...])
        t = (tt)[..., None]
        if w == 0:
            A = torch.exp(-k * tt)
        else:
            A = torch.exp(k * (w * w * k / (4.0) - t)) \
                * 0.5 * torch.erfc(-t / w + w * k / (2.0))
        if model_coh:
            coh = torch.exp(-0.5 * (tt / w) * (tt / w))
            coh = coh[:, :, None].repeat((1, 1, 3))
            coh[..., 1] *= (-tt * exp_half / w)
            coh[..., 2] *= (tt * tt / w / w - 1)
            A = torch.cat((A, coh), dim=-1)

        X, fit, res = lstsq(A, self.data)
        self.done_eval = True
        self.c = X
        self.model = fit
        self.residuals = res
        return X, fit, res

    def fit_func(self, x):
        ds = self.dataset
        disp_coefs = x[:self.disp_poly_deg]
        w = float(x[self.disp_poly_deg])
        taus = x[self.disp_poly_deg + 1:]
        t_zeros = np.poly1d(disp_coefs)(ds.wavenumbers)
        tt = np.subtract.outer(ds.t, t_zeros).T
        c, model, res = self.eval_torch(tt, w, taus, True)
        return res.cpu().numpy().ravel()

    def start_fit(self,
                  w,
                  taus,
                  fix_last_tau=False,
                  fix_width=False,
                  fix_disp=False):
        ds = self.dataset
        time_zeros = self.zero_func(ds.wavenumbers)
        disp_guess = np.polyfit(ds.wavenumbers, time_zeros, self.disp_poly_deg)
        x0 = np.hstack((disp_guess, w, taus))
        idx = np.ones_like(x0, dtype='bool')
        if fix_last_tau:
            idx[-1] = False
        if fix_width:
            idx[self.disp_poly_deg] = False
        if fix_disp:
            idx[:self.disp_poly_deg] = False

        start_guess = x0[idx]

        def fix_func(x):
            x0[idx] = x
            return self.fit_func(x0)

        bounds = np.array([(-np.inf, np.inf)] * len(x0))
        bounds[self.disp_poly_deg:, 0] = 0
        bounds = bounds[idx, :]
        x = least_squares(fix_func, start_guess, bounds=bounds.T)
        return x, x0
