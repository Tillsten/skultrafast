# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:29:52 2012

@author: Tillsten
"""

import numpy
import pylab
from numpy.fft import rfft
from numpy.fft.helper import fftshift, fftfreq

'''This module contains a few very simple routines to compute and
optionally plot spectral diagnostics based on the discrete Fourier
transform (DFT). Note that, in general, the results will be meaningful only for
stationary processes sampled at regular time intervals.'''

def DftPowerSpectrum(x, dt = 1, norm = False, doplot = False):
    '''Compute power spectrum (squared modulus of discrete Fourier
    transform) of 1-D vector x, assumed to be sampled regularly with
    sampling interval dt, and the corresponding frequency array in
    physical units (postive frequencies only). If norm is True, the
    power specturm is "normalised", i.e. multiplied by 4, so that a
    sinusoid with semi-amplitude A gives rise to a peak of height A**2
    in the power spectrum.'''
    n = x.size
    amp = abs(rfft(x)) 
    ps = amp**2 / float(n)
    if norm == True: ps *= 4 / float(n)
    freq = numpy.arange(n/2+1) / float(n*dt)
    if doplot == True:
        pylab.plot(freq, ps)
        pylab.xlabel('Frequency')
        pylab.title('DFT power spectrum')
        pylab.ylabel('Power')
    return ps, freq

def AcfPeriodogram(x, dt = 1, norm = False, doplot = False, \
                       smooth = False, box = 0.01):
    '''Compute periodogram (power spectrum of ACF) of a 1-D vector x,
    assumed to be sampled regularly with sampling interval dt, and the
    corresponding frequency array in physical units (postive
    frequencies only). The ACF is computed up to lags of N/4 where N
    is the length of the input array. If smooth is True, the ACF is
    multiplied by sinc(pi/box) before taking the power spectrum. This
    is equivalent to smoothing the power specturm by convolving it a
    top-hat function of width (box/dt) in the frequency domain.'''
    maxl = min(len(x), max(len(x)/4, 50))
    lag, corr, line, ax = pylab.acorr(x, maxlags = maxl)
    if smooth == True:
        corr *= sinc(numpy.pi/box)
    pgram, freq = DftPowerSpectrum(corr, dt, doplot = False)
    if doplot == True:
        pylab.plot(freq, pgram)
        pylab.xlabel('Frequency')
        pylab.ylabel('Power')
        if smooth == True:
            pylab.title('ACF smoothed periodogram')
        else:
            pylab.title('ACF periodogram')
    return pgram, freq
    


def calc_C(t, x, omega):    
    '''Computes the Schuster periodogram, applying small correction
    for non-orthogonality. Also returns the ML estimates of the
    amplitudes.'''
    n = x.size
    nw = omega.size
    R = numpy.zeros(nw)
    I = numpy.zeros(nw)
    for i in numpy.arange(nw):
        R[i] = (x * numpy.cos(omega[i] * t)).sum()
        I[i] = (x * numpy.sin(omega[i] * t)).sum()
    X = numpy.sin(n*omega) / numpy.sin(omega)
    c, s = n + X, n - X
    return R**2/c + I**2/s, [2 * R / float(n), 2 * I / float(n)]

def calc_L(C, sigma = None, x = None):
    if sigma != None:
        L = numpy.exp(C / sigma**2)
    else:
        Y = (x**2).sum()
        n = x.size
        L = (1 - 2 * C / Y)**(1-n/2.)
    return L 

def calc_amp_sigma(t, x, omega, C, amps):
    nw = C.size
    n = t.size
    est_var = numpy.zeros(nw)
    amp_err = numpy.zeros(nw)
    for j in numpy.arange(nw):
        model = amps[0][j] * numpy.cos(omega[j] * t) + \
            amps[1][j] * numpy.sin(omega[j] * t)
        est_var[j] = ((x-model)**2).sum() / float(n)
    amp = numpy.sqrt(amps[0]**2+amps[1]**2)
    sigma = numpy.sqrt(est_var)
    return amp, sigma

def est_param(C, omega, amp, sigma, N):
    i = numpy.argmax(C)
    omega_est = omega[i]
    amp_est = amp[i]
    sigma_est = sigma[i]
    omega_err = (sigma_est / amp_est) * numpy.sqrt(48/float(N)**3)
    amp_err = sigma_est / numpy.sqrt(float(N))
    sigma_err = numpy.sqrt(2 / float(N-2-4))
    return omega_est, omega_err, amp_est, amp_err, sigma_est, sigma_err

def best_fit(t, C, omega, amps):
    i = numpy.argmax(C)
    model = amps[0][i] * numpy.cos(omega[i] * t) + \
        amps[1][i] * numpy.sin(omega[i] * t)
    return model

def calc_PSD(L, C, sigma):
    return 2 * (sigma**2 + C) * L

def analyse(time, data_in, omega = None, doplot = False):
    data = scipy.copy(data_in)
    themean = data.mean()
    data -= themean
    N = data.size
    dt = time[1] - time[0]
    dft_ps, freq = dft.DftPowerSpectrum(data, dt = dt)
    dft_omega = 2 * numpy.pi * freq
    if omega == None: omega = numpy.r_[0.0001:numpy.pi:0.0001]
    C , amps = calc_C(time, data, omega)
    amp, sigma = calc_amp_sigma(time, data, omega, C, amps)
    L = calc_L(C, x = data)
    dw = omega[1] - omega[0]
    L = L / (L*dw).sum()
    PSD = calc_PSD(L, C, sigma)
    params = est_param(C, omega, amp, sigma, N)
    omega_est, omega_err, amp_est, amp_err, sigma_est, sigma_err = params
    l = (omega >= (omega_est - 2 * omega_err)) * \
        (omega <= (omega_est + 2 * omega_err))
    P95 = (L[l]*dw).sum()
    model = best_fit(time, C, omega, amps) + themean
    residuals = data - model

    if doplot != False:
        print 'P(95):', P95
        print 'Estimated angular frequency:', omega_est, '+/-', omega_err
        print 'Estimated amplitude:', amp_est, '+/-', amp_err
        print 'Estimated noise st.dev:', sigma_est, '+/-', sigma_err
        ee = pu.dofig(1, 1, 2, aspect = 1)
        ax1 = pu.doaxes(ee, 1, 2, 0, 0)
        pylab.title('Time series')
        pylab.plot(time, data, 'k-')
        pylab.plot(time, model, 'r-')
        pylab.ylabel('signal')
        ax2 = pu.doaxes(ee, 1, 2, 0, 1, sharex = ax1)
        pylab.plot(time, residuals, 'k-')
        pylab.ylabel('residuals')
        pylab.xlabel('time')
        pylab.xlim(time.min(), time.max())
        ee = pu.dofig(2, 1, 2, aspect = 1)
        ax1 = pu.doaxes(ee, 1, 2, 0, 0)
        pylab.plot(dft_omega , dft_ps, 'wo', mec = 'k')
        pylab.plot(omega, C, 'k-')
        pylab.plot(omega, 4*sigma**2, 'r-')
        pylab.ylabel('Power')
        pylab.legend(('DFT', 'Schuster','Noise'))
        ax2 = pu.doaxes(ee, 1, 2, 0, 1, sharex = ax1)
        pylab.plot(omega, PSD, 'k-')
        pylab.ylabel('Bayes PSD')
        pylab.xlabel('angular frequency')
        pylab.xlim(omega.min(), omega.max())

    return omega, L, C, sigma, params, P95, model
    
import scipy
import scipy.linalg
import pylab
#import mpfit

mach = scipy.MachAr()
small = 10 * mach.eps

def sinefit(time, data, err = None, pmin = None, pmax = None, \
                nper = 500, return_periodogram = False, doplot = False):
    """ML sine curve fit. Period by brute force, other pars linear.
    per, amp, phase, dc = sinefit(x, y, [err, pmin, pmax, nper])"""
    npts = len(time)
    if pmin is None:
        w = scipy.sort(time)
        dt = w[1:] - w[:npts-1]
        tstep = scipy.median(dt)
        pmin = 2 * tstep
    if pmax is None: pmax = (time.max() - time.min()) / 2.
    lpmin,lpmax = scipy.log10([pmin,pmax])
    lpers = scipy.r_[lpmin:lpmax:nper*1j]
    pers = 10.0**lpers
    if err == None:
        z = scipy.ones(npts)
        err = scipy.zeros(npts)
        mrk = '.'
    else:
        if len(err) is len(data):
            z = 1.0 / err**2
            mrk = '.'
        else:
            z = scipy.ones(npts)
            err = scipy.zeros(npts)
            mrk = '.'
    sumwt = z.sum()
    chi2_0 = scipy.sum((data-scipy.mean(data))**2*z)
    p_w = scipy.zeros(nper)
    a_w = scipy.zeros(nper)
    p_max = -1.0e20

    for i in scipy.arange(nper):
        arg = 2 * scipy.pi * time / pers[i]
        cosarg = scipy.cos(arg)
        sinarg = scipy.sin(arg)
        a = scipy.matrix([[scipy.sum(sinarg**2*z), scipy.sum(cosarg*sinarg*z), \
                               scipy.sum(sinarg*z)], \
                              [0, scipy.sum(cosarg**2*z), scipy.sum(cosarg*z)], \
                              [0, 0, sumwt]])
        a[1,0] = a[0,1]
        a[2,0] = a[0,2]
        a[2,1] = a[1,2]
        a[abs(a) < small] = 0.
        if scipy.linalg.det(a) < small: continue
        b = [scipy.sum(data*sinarg*z), scipy.sum(data*cosarg*z), \
                 scipy.sum(data*z)]
        c = scipy.linalg.solve(a,b)  
        amp = (c[0]**2+c[1]**2)**(0.5)
        a_w[i] = amp
        phase = scipy.arctan2(c[1],c[0])
        dc = c[2]
        fit = amp * scipy.sin(arg + phase) + dc
        p_w[i] = (chi2_0 - scipy.sum((data-fit)**2 * z)) / chi2_0
        if p_w[i] > p_max:
            p_max = p_w[i]
            oper = pers[i]
            oamp = amp
            ophase = phase
            odc = dc
            ofit = fit

    if doplot == False: 
        if return_periodogram == False: return oper, oamp, ophase, odc
        else: return oper, oamp, ophase, odc, pers, p_w, a_w, chi2_0

    pylab.close('all')
    pylab.figure(1, figsize = (6,7), edgecolor = 'w')
    pylab.subplot(311)
    pylab.errorbar(time, data, err, fmt = 'k' + mrk, capsize = 0)
    pylab.xlabel('x')
    pylab.ylabel('y')
    np = (time.max()-time.min()) / oper
    if np < 20:
        x = scipy.r_[time.min():time.max():101j]
        pylab.plot(x, oamp * scipy.sin(2 * scipy.pi * x / oper + ophase) + odc, 'r')
    pylab.xlim(time.min(), time.max())
    pylab.subplot(312)
    pylab.loglog()
    pylab.axvline(oper, c = 'r')
    pylab.plot(pers, p_w, 'k-')
    pylab.xlabel('period')
    pylab.ylabel('reduced chi2')
    pylab.xlim(pers.min(), pers.max())
    pylab.subplot(313)
    ph = (time % oper) / oper
    pylab.errorbar(ph, data, err, fmt = 'k' + mrk, capsize = 0)
    x = scipy.r_[0:oper:101j]
    y = oamp * scipy.sin(2 * scipy.pi * x / oper + ophase) + odc
    pylab.plot(x/oper, y, 'r')
    pylab.xlim(0,1)
    pylab.xlabel('phase')
    pylab.ylabel('y')

    if return_periodogram == False: return oper, oamp, ophase, odc
    else: return oper, oamp, ophase, odc, pers, p_w, a_w, chi2_0