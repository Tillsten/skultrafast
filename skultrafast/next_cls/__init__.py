import numpy as np
from two_d_nextcls import Sigma1NN, Sigma2NN, SigmaInfNN
from scipy.constants import c

c_cmps = c*100/1e12
print(c_cmps)


def CAFE(amps, taus, fwhm):
    """Calculate the CAFE parameter."""
    amps = np.asarray(amps)
    taus = np.asarray(taus)

    assert(len(amps) == 3)
    assert(len(taus) in [2, 3])
    assert(np.all(taus >= 0))

    if len(taus) == 2:
        taus = np.append(taus, 1e6)
    idx = np.argsort(taus)
    taus = taus[idx]
    amps = amps[idx]
    #print(amps, taus)
    s = rescale_timescale(amps, taus, fwhm)

    log_s = np.log10(s)
    # print(log_s)
    if amps[1] == 0:
        log_s[1] == 0
    x = [amps[0], log_s[0], amps[1], log_s[1], amps[2]]
    print(x)
    sig1 = Sigma1NN(x)
    sig2 = Sigma2NN(x)

    sig_inf = SigmaInfNN(x)
    delta2 = 10**sig2/(2*np.pi*c_cmps*taus[1])
    print(sig2, delta2)
    return 10**sig1, delta2, 10**sig_inf


def rescale_timescale(amps, taus, fwhm):
    """Rescale the timescale to the FWHM of the correlation function."""
    return fwhm * np.sqrt(amps/amps.sum())*taus*2*np.pi*c_cmps


def delta1(amps, taus, fhwm):
    pass


def deltas(sigmas, taus):
    return sigmas/taus


def delta_inf(sigma_inf, fwhm):
    return fwhm/2*np.sqrt(sigma_inf/2/np.log(2))


CAFE(amps=[0.5, 0.2, 0], taus=[1, 5], fwhm=50)[1]
# %%


def line_shape_function(t, T2, taus, deltas):
    t = np.asarray(t)
    T2 = np.asarray(T2)
    taus = np.asarray(taus)
    deltas = np.asarray(deltas)

    assert(len(taus) == len(deltas))
    a = t/T2
    a += np.sum(deltas**2 * (taus**2 * (np.exp(-t/taus) - 1) + t*taus))
    return a


line_shape_function([3], [0.2], [1], [0.2])
# %%
