"""
Sympy notebook with the convolution
===================================
In this notebook we use *sympy*, a CAS for python, to calculate the convolution of an
gaussian with an one-sided decaying exponential. *sympy* is not an requirement of
*skultrafast*, there it is possible that it must be manually installed.
"""
# %%
import sympy
from sympy import (symbols, Heaviside, exp, sqrt, oo, integrate, simplify,
                   special,  plot, pi, init_printing, solve)
init_printing()
# %%
# First we need to define sympy symbols.

A, t, ti = symbols('A t ti', real=True)
tau, sigma = symbols('tau sigma', positive=True)
step = Heaviside

# %% [rsT]
# Define :math:`y=A\exp(-t/\tau)` and the gaussian IRF.

# %%
y = step(t) * A * exp(-t / tau)
y

# %%
irf = 1 / sqrt(2 * pi * sigma**2) * exp(-t**2 / (2 * sigma**2))
irf

# %%
# Next, we will calculate the covolution integral.

# %%
func = integrate((irf.subs(t, t - ti) * y.subs(t, ti)), (ti, -oo, oo))
func = simplify(func)
func

# %%
# Rewirte the `erf` with the `erfc` function:
erfc, erf = special.error_functions.erfc, special.error_functions.erf
func2 = func.rewrite(erfc)
func2

# %% [markdown]
# Plot the result to it makes sense:
plot(func2.subs(sigma, 0.2).subs(tau, 2).subs(A, 1), (t, -1, 10))

# %%
# Normalized derivatives of a gaussian
# ------------------------------------
# Used to model coherent contributions.

irf, irf.diff(t, 1), irf.diff(t, 2)

# %%
plot(irf.diff(t).subs(sigma, 0.2).subs(tau, 2), (t, -1, 1))
plot(irf.diff(t, 2).subs(sigma, 0.2).subs(tau, 2), (t, -1, 1))

# %%
# Find the maxima and evalute the positions
sol = solve(Eq(irf.diff(t, 2), 0), t)
norm1 = dirf.subs(t, sol[0])
sol, norm1

# %%
sol = solve(Eq(irf.diff(t, 3), 0), t)
norm2 = irf.diff(t, 2).subs(t, sol[0])
sol, norm2

# %%
nf1, nf2 = irf.diff(t, 1) / norm1, irf.diff(t, 2) / norm2

# %%
plot(
    nf1.subs(sigma, 0.2).subs(tau, 2),
    nf2.subs(sigma, 0.2).subs(tau, 2), (t, -1, 1))

# %%
nf1

# %%
nf2
