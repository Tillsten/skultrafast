speed_of_light = 299792458.0

def freq_grid(N, step_fs):
    """
    Calculates the frequency grid in cm-1.

    Parameter
    """
    fr = np.fft.fftfreq(y.size, step_fs) * 1e15
    wn = (fr / speed_of_light) / 1e2
    return wn

def load_and_bin(fname):
    a = np.load('../test_daten//bla33_%d.npy' % i)[:, :]
    pd1, pd2, pyro = a[:, -3:].T
    pd1 = detrend(pd1)
    pd1_state = (pd1 - pd1.mean()) > 0
    pd2_state = (pd2 - pd2.mean()) > 0
    pyro = detrend(pyro)
    counts = np.hstack((0, np.diff(pd1_state).cumsum())) + np.hstack((0, np.diff(pd2_state).cumsum()))
    tau = counts * HeNe_period_fs / 4.
    step = HeNe_period_fs
    pre_bins = np.arange(0, tau.max(), step)
    m, _, _ = binned_statistic(tau, pyro, 'mean', bins=pre_bins)
    # m -= m.mean()
    binned_probe = binned_statistic(tau, a[:, :16].T,
                                    'mean', bins=pre_bins)[0]
    k0, const_phase = phase_ifg(m[:], step=step, j=1)
