from typing import Callable
import attr
import numpy as np
from scipy.constants import speed_of_light as c
from skultrafast.unit_conversions import invps2cm, cm2invps
from scipy.interpolate import interp2d


def response_functions(g, coords, omega, domega, two_level=False):
    T1, t2, T3 = coords
    anh = 5
    if two_level:
        R_r = np.exp(-1j*omega*(-T1+T3))*np.exp(-g(T1)+g(t2) -
                                                g(T3)-g(T1+t2)-g(t2+T3)+g(T1+t2+T3))
        R_nr = np.exp(-1j*omega*(-T1+T3))*np.exp(-g(T1)-g(t2) -
                                                 g(T3)+g(T1+t2)+g(t2+T3)-g(T1+t2+T3))
    else:
        gT1 = g(T1)
        gt2 = g(t2)
        gT3 = gT1.T
        gT1t2 = g(T1+t2)
        gt2T3 = gT1t2.T
        ga = g(T1+t2+T3)
        pop = (2-2*np.exp(-1j*anh*T3))
        osc = np.exp(-1j*omega*(-T1+T3))
        R_r = osc*np.exp(-gT1+gt2-gT3-gT1t2-gt2T3+ga)*pop
        R_nr = osc*np.exp(-gT1-gt2-gT3+gT1t2+gt2T3-ga)*pop

    R_r[:, 0] *= 0.5
    R_r.T[:, 0] *= 0.5
    R_nr[:, 0] *= 0.5
    R_nr.T[:, 0] *= 0.5
    return R_r, R_nr


@attr.define
class KuboFitter:
    t1: np.ndarray
    t2: np.ndarray
    probe_wn: np.ndarray
    rot_frame_freq: float
    ifgr: np.ndarray

    kubo_fcn: Callable

    def resample(self):
        freqs = self.t1
        out = np.zeros((self.t2.size, self.t1.size, self.t1.size))

        for i, wt in enumerate(self.t2):
            intp = interp2d(self.t1, self.probe_wn, self.ifgr[i, :, :], kind='cubic')
            out[i, :, :] = intp()

    def simulate_ifg(self, anh, kubo_params):
        def f(t):
            return self.kubo_fcn(t, **kubo_params)

        response_functions(f, pass)
