
# %%
import numba
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import c
from sympy import OmegaPower

cm2invps = c*1e-12
invps2cm = 1/cm2invps
1/(1700 * cm2invps)

# %%
Δω = 5
τ = 1
ω = 0
Delta = 5

t2 = 0.3
dt = 0.05
n_t = 256
n_zp = n_t*2
t = np.arange(0, n_t)*dt

λ = 1/τ

two_level = True

print(f'Δω / λ = {Δω / λ}')
print(f'Δω  = {Δω} ps^-1')
print(f'λ  = {λ} ps^-1')


# %%


T1, T3 = np.meshgrid(t, t,)


@numba.jit
def g(t, dom, lam):
    return (dom / lam)**2 * (np.exp(-lam*t)-1 + lam*t)


fig, ax = plt.subplots()
ax.plot(t, np.exp(-g(t, dom=5, lam=1/5)))
ax.set(xlim=(0, 2), xlabel="t / ps")

# %%

coods = tuple(T1, )


@numba.jit
def response_functions(, g, **kwargs):
    coords = T1, t2, T3
    anh = 5
    if two_level:
        R_r = np.exp(-1j*ω*(-T1+T3))*np.exp(-g(T1)+g(t2) -
                                            g(T3)-g(T1+t2)-g(t2+T3)+g(T1+t2+T3))
        R_nr = np.exp(-1j*ω*(-T1+T3))*np.exp(-g(T1)-g(t2) -
                                             g(T3)+g(T1+t2)+g(t2+T3)-g(T1+t2+T3))
    else:
        gT1 = g(T1)
        gt2 = g(t2)
        gT3 = gT1.T
        gT1t2 = g(T1+t2)
        gt2T3 = gT1t2.T
        ga = g(T1+t2+T3)
        pop = (2-2*np.exp(-1j*anh*T3))

        R_r = np.exp(-1j*ω*(-T1+T3))*np.exp(-gT1+gt2-gT3-gT1t2-gt2T3+ga)*pop
        R_nr = np.exp(-1j*ω*(T1+T3))*np.exp(-gT1-gt2-gT3+gT1t2+gt2T3-ga)*pop
    R_r[:, 0] *= 0.5
    R_r.T[:, 0] *= 0.5
    R_nr[:, 0] *= 0.5
    R_nr.T[:, 0] *= 0.5
    return R_r, R_nr


def response_to_spec(R_r, R_nr):
    fR_r = np.fft.fft2(R_r, s=(n_zp, n_zp))
    fR_nr = np.fft.fft2(R_nr, s=(n_zp, n_zp))


# %%

# %%
fig, ax = plt.subplots(2, sharex='all', sharey='all', figsize=(4, 4))
ax[0].pcolormesh(R_r.real, rasterized=True)
ax[1].pcolormesh(R_nr.real, rasterized=True)
#plt.setp(ax[1], xlim=(0, 50), ylim=(0, 50))

# %%
fig, ax = plt.subplots(2, sharex='all', sharey='all',
                       figsize=(8, 8), subplot_kw=dict(aspect=1), dpi=150)
fR_r = np.fft.fft2(R_r, s=(n_zp, n_zp))
fR_nr = np.fft.fft2(R_nr, s=(n_zp, n_zp))

ax[0].contourf(np.fliplr(np.fft.fftshift(fR_r.real)), 20, cmap='bwr')
ax[1].contourf(np.fft.fftshift(fR_nr.real), 20, cmap='bwr')
m = t.size
plt.setp(ax[1], xlim=(t.size-m/2, t.size+m/2), ylim=(t.size-m/2, t.size+m/2))

# %%
R = np.fft.fftshift(np.real(fR_r+fR_nr))
#R += np.flipud(np.fliplr(R))

fig, ax = plt.subplots()
ax.contourf(np.fft.fftshift(np.fliplr(fR_r)+fR_nr).real, 20, cmap='bwr')
plt.setp(ax, xlim=(t.size-m/2, t.size+m/2), ylim=(t.size-m/2, t.size+m/2))

# %%

# # %%
# import sympy as s
# from sympy.functions import Heaviside
# T1, T2, T3, t = s.symbols('T1 T2 T3 t', real=True)
# dw, w, lam = s.symbols('domega omega lambda', positive=True)
# g = s.Lambda(t, (dw/lam)**2 * (s.exp(-lam*t)-1+lam*t))

# R_r =  s.exp(-g(T1)+g(T2)-g(T3)-g(T1+T2)-g(T2+T3)+g(T1+T2+T3))
# R_nr = s.exp(-g(T1)-g(T2)-g(T3)+g(T1+T2)+g(T2+T3)-g(T1+T2+T3))
# rot = s.exp(-1j*w*(-T1+T3))
# s.simplify(R_r+R_nr)
# 3#.integrals.fourier_transform(Heaviside(T1)*R_r)
# # %%
# from sympy.integrals.transforms import fourier_transform
# fourier_transform(Heaviside(T2)*R_r, T2, w)
# # %%

# %%

# %%
