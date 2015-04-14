# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:10:26 2015

@author: Tillsten
"""

import numpy as np
import emcee
import lmfit

def create_prior(params):
    """
    emccee uses a uniform prior for every variable.
    Here we create a functions which checks the bounds
    and returns np.inf if a value is outside of its
    allowed range. WARNING: A uniform prior may not be
    what you want!
    """
    none_to_inf = lambda x, sign=1: sign*np.inf if x is None else x
    lower_bounds = np.array([none_to_inf(i.min, -1) for i in params.values() if i.vary])
    upper_bounds = np.array([none_to_inf(i.max, 1) for i in params.values() if i.vary])
    def bounds_prior(values):
        values = np.asarray(values)        
        is_ok = np.all((lower_bounds < values) & (values < upper_bounds))
        return 0 if is_ok else -np.inf
    return bounds_prior

def create_lnliklihood(mini, use_t=False, sigma=None):
    """create a normal-likihood from the residuals"""
    def lnprob(vals, sigma=sigma):
        for v, p in zip(vals, [p for p in mini.params.values() if p.vary]):
            p.value = v
        residuals = mini.userfcn(mini.params)
        if not sigma:
            #sigma is either the error estimate or it will
            #be part of the sampling.
            sigma = vals[-1]
            if sigma <= 0:
                return -np.inf
        if use_t:
            from scipy.stats import  t, norm
            val = t.logpdf(residuals, df=5, scale=sigma).sum()
            
        else:
            val = -0.5*np.sum(np.log(2*np.pi*sigma**2) + (residuals/sigma)**2)
        return val
    return lnprob

def starting_guess(mini, estimate_sigma=True):
    """
    Use best a fit as a starting point for the samplers.
    If no sigmas are given, it is assumed that
    all points have the same uncertainty which will
    be also part of the sampled parameters.
    """
    vals = [i.value for i in mini.params.values() if i.vary]
    if estimate_sigma:
        vals.append(mini.residual.std())
    return vals

def create_all(mini, use_t=False, sigma=None):
    """
    creates the log-poposterior function from a minimizer.
    sigma should is either None or an array with the
    1-sigma uncertainties of each residual point. If None,
    sigma will be assumed the same for all residuals and
    is added to the sampled parameters.
    """
    sigma_given = not sigma is None
    lnprior = create_prior(mini.params)
    lnprob = create_lnliklihood(mini, use_t=use_t, sigma=sigma)
    guess = starting_guess(mini, not sigma_given)
    if sigma_given:
        func = lambda x: lnprior(x[:]) + lnprob(x)
    else:
        func = lambda x: lnprior(x[:-1]) + lnprob(x)
    return func, guess
    
    
def get_errors(m, steps=2000, use_t=False):
    lnfunc, guess = create_all(m, use_t=use_t)
    nwalkers, ndim = 40, len(guess)
    p0 = emcee.utils.sample_ball(guess, 0.2*np.array(guess), nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnfunc)
    steps = steps
    sampler.run_mcmc(p0, steps)
    para = m.params
    not_fixed = [p for p in para if para[p].vary]
    for i, p in enumerate(not_fixed):
        m.params[p].value = guess[i]
    return sampler

#create lnfunc and starting distribution.