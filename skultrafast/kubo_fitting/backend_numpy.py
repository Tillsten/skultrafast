from dataclasses import dataclass, field
from typing import Protocol
from numpy import exp, pi, sqrt, zeros, ndarray, asarray, array, fft, rot90, flip, roll


__all__ = ['fftshift2d', 'g', 'response_functions_1D', 'response_functions_2D']


def fftshift2d(x: ndarray) -> ndarray:
    """
    Parameters
    ----------
    x : ndarray
        The input array.

    Returns
    -------
    ndarray
        The shifted array, at least two dims.

    """
    return fft.fftshift(x, axes=(-1, -2))


def g(t: ndarray, dom: ndarray = 5, lam: ndarray = array([1])) -> ndarray:
    """
    Generalized kubo function of the form:

    .. math::
        g(t) = \\frac{\\mathrm{dom}}{\\lambda^2} \\left( e^{-\\lambda t} - 1 + \\lambda t \\right

    Parameters
    ----------
    t : ndarray
        The time points to evaluate the kubo function.
    dom : ndareray, optional
        Amplitude of the decay. Has to be the same shape as lam, by default 5.
    lam : ndarray, optional
        The spectral density values, by default array([1]).

    Returns
    -------
    ndarray
        The correlation function evaluated at the given time points.

    """
    lam = asarray(lam)
    dom = asarray(dom)
    if t.ndim == 1:
        lam = lam[:, None]
        dom = dom[:, None]
    else:
        lam = lam[:, None, None, None]
        dom = dom[:, None, None, None]
    t = t[None, ...]
    s = (dom / lam)**2 * (exp(-lam * t) - 1 + lam*t)
    return s.sum(0)


def response_functions_1D(T1: ndarray,
                          omega: float = 0,
                          dom: ndarray = array([5]),
                          lam: ndarray = array([1])) -> ndarray:
    """
    Generate the response function for 1D IR.

    Parameters
    ----------
    T1 : ndarray
        The time points for the 1D IR.
    omega : float, optional
        The frequency shift, by default 0.
    dom : ndarray, optional
        The spectral density domain parameter, by default 5.
    lam : ndarray, optional
        The spectral density values, by default array([1]).

    Returns
    -------
    ndarray
        The response function for 1D IR.

    """
    lam = asarray(lam)
    osc = exp(-1j * omega * (-T1))
    FID = osc * exp(-g(T1, dom, lam))
    FID[0] *= 0.5
    return FID


def response_functions_2D(
        T1: ndarray,
        T2: ndarray,
        T3: ndarray,
        omega: float,
        dom: ndarray = array([5]),
        anh: float = 5,
        lam: ndarray = array([1]),
) -> tuple[ndarray, ndarray]:
    """
    Generate the response function for 2D IR.
    We assume T1 and T3 are the same for performance reasons.

    Parameters
    ----------
    T1 : ndarray
        The time points for the first dimension of 2D IR (pulse seperation).
    T2 : ndarray
        The time points for waiting times.
    T3 : ndarray
        The time points for the second dimension of 2D IR (spectrum).
    omega : float
        The frequency shift.
    dom : float, optional
        Delta omega, the
    anh : float, optional
        The anharmonicity parameter, by default 5.
    lam : ndarray, optional
        The kubo decay rates, by default np.array([1]).

    Returns
    -------
    tuple[ndarray, ndarray]
        The two response functions for 2D IR.

    """
    lam = asarray(lam)
    T1 = T1[None, :, :]
    T2 = T2[:, None, None]
    T3 = T3[None, :, :]

    gT1 = g(T1, dom, lam)
    gt2 = g(T2, dom, lam)
    gT3 = gT1.transpose(0, 2, 1)
    gT1t2 = g(T1 + T2, dom, lam)
    gt2T3 = gT1t2.transpose(0, 2, 1)
    ga = g(T1 + T2 + T3, dom, lam)
    pop = (2 - 2 * exp(-1j * anh * T3))
    if omega != 0:
        osc = exp(-1j * omega * (-T1 + T3))
        osc2 = exp(-1j * omega * (T1 + T3))
    else:
        osc = 1
        osc2 = 1

    # Lines fromt the 2D IR book example
    # R_r=exp(-1i*w_0.*(-T1+T3)).*exp(-g(T1)+g(t2)-g(T3)-g(T1+t2)-g(t2+T3)+g(T1+t2+T3)).*(2-2.*exp(-sqrt(-1)*Delta.*T3));
    # R_nr=exp(-1i*w_0.*(T1+T3)).*exp(-g(T1)-g(t2)-g(T3)+g(T1+t2)+g(t2+T3)-g(T1+t2+T3)).*(2-2.*exp(-sqrt(-1)*Delta.*T3));
    # Reordered for better numerical stability.

    R_r = osc * exp(-gT1 - gT3 - gT1t2 - gt2T3 + gt2 + ga) * pop
    R_nr = osc2 * exp(-(gT1 + gt2 + gT3 + ga) + gT1t2 + gt2T3) * pop  # type: ignore

    # Multiply by 0.5 to account for the fact that we are only using half of the
    # response function. This is because we are only using half of the spectrum
    # and half of the pulse seperation.

    R_r[:, :, 0] *= 0.5
    R_r[:, 0, :] *= 0.5
    R_nr[:, :, 0] *= 0.5
    R_nr[:, 0, :] *= 0.5
    return R_r, R_nr


def response_to_spec_2D(R_r, R_nr):
    # R_nr, R_r = response_functions(T1, T2, T3, ω)
    fR_nr, fR_r = fft.fft2(R_nr), fft.fft2(R_r)
    fR_nr, fR_r = fftshift2d(fR_nr), fftshift2d(fR_r)
    R = rot90(fR_nr, 2, axes=(1, 2))
    R += flip(roll(fR_r, -1, axis=1), axis=(2, ))
    R = R.real
    return R


def response_to_spec_1D(R):
    fR = fft.fft(R)
    fR = fft.fftshift(fR)
    R = R.real
    return R


@dataclass
class KuboBackend(Protocol):
    T1: ndarray
    T2: ndarray
    T3: ndarray

    def g(self, t, dom, lam) -> ndarray:
        ...

    def response_functions_1D(self, omega=0, dom=5, lam=array([1])) -> ndarray:
        ...

    def response_functions_2D(self,
                              omega,
                              dom=5,
                              anh=5,
                              lam=array([1])) -> tuple[ndarray, ndarray]:
        ...

    def response_to_spec_2D(self, R_r, R_nr):
        ...
