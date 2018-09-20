import numpy as np
import numexpr as ne


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
    y = ne.evaluate("np.exp(k * (ws * ws * k / (4.0) - t)) * 0.5 * erfc(-t / ws + ws * k / (2.0))")