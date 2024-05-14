"""
Signal processing of a 2D-IR experiment
=======================================

In this example we will show how to process a 2D-IR signal, starting at the raw data.
The setup uses a pump-probe configuration with a delay stage for the pump pulse.
A AOM is used to generate the double pulse and adjust the phases.

We use a 4-step phase cycle to reduce scatter and background. Also we use a
roating frame to recude the number of necessary data points.
"""

# %%
# First the necessary imports.
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from skultrafast.plot_helpers import enable_style

enable_style()

plt.rcParams["pcolor.shading"] = "gouraud"
# %% The data is stored in a HDF5 file. We will use the h5py library to read the
# data. Not that file also contains partly processed data, however we will start
# from the raw data.

fpath = r"D:\boxup\skultrafast\skultrafast\examples\data\24-02-22 12_07 2D SCN5_pXylol - Copy (2).messpy"

with h5.File(fpath, "r") as f:
    # The data is stored in the group 'data'. We will extract the data and the axes.
    t1 = f["t1"][:]  # Inter Pump-pump delay axis
    t2 = f["t2"][:]  # Pump-Probe delay axis, also called waiting time
    wl = f["wl"][:]  # Probe Wavelength axis, given by the spectrometer and the detector


# %%
# Our setup uses three detector lines, two probe lines and one reference line.
# Here, we will only use one Probe line. Each sub-group contains the data for
# a single waiting time. Each singl waiting time contains the data for all
# scans.

with h5.File(fpath, "r") as f:
    frames = f["frames/Probe1"]  # The data itself from the first probe.
    n_wt = len(frames)

    # Lets look at the data of the second waiting time and the first scan.
    # This should correspond to 0.3 ps.
    data = frames["10"]["0"][:]
    n_scans = len(frames["2"])

# %%
# Lets compara the shape of the data with the shape of the axes.
print("data", data.shape, "t1", t1.shape, "wl", wl.shape)

# %% Notice that the second axis of the data is four times the length of the t1
# axis. This is due to the 4-step phase cycle, where we record the signal for
# each delay in four different phase cycles. Currently, the data are the raw
# pixel values averaged over some repeated measurements. Looking at the data, we
# do not see much, since the pump-induced change is very small compared to the
# probe.

fig, ax = plt.subplots()
ax.imshow(data, aspect="auto")

fig, (ax, ax2) = plt.subplots(2, figsize=(4, 3), constrained_layout=True)
n = 58
ax.plot(data[n], label="Channel %d, %d nm" % (n, wl[n]))

ax.legend()
ax2.plot(wl, data[:, 0], label="Channel 1, t1 = %.1f fs" % t1[0])
ax2.plot(wl, data[:, 1], label="Channel 1, t1 = %.1f fs" % t1[0])
ax2.plot(wl, data[:, 2], label="Channel 1, t1 = %.1f fs" % t1[0])
ax2.axvline(wl[58], color="k", ls="--")
ax2.legend()
ax2.set_xlabel("Wavelength [nm]")
fig.supylabel("Pixel Count [a.u.]")

# %%
# In the used cycling scheme, the relative phase of the pump pulse is changed in
# each cycle, hence the signal is given by the difference of the signal in the
# two phases. We also modulate the phase relative to the probe pulse by pi.
# So the signal is given by the difference of the signal in the two phases and
# summed up over two the two phases. This gives us the inferogram.

# Sig = (S1 - S2) + (S3 - S4)

# Note that this is the linear approximation of the signal. The actual signal
# is the log of the ratio of the signal in the two phases.
sig = np.log(data[:, ::4] / data[:, 1::4]) #+ np.log1p((data[:, 2::4] / data[:, 3::4]))

# This also has to be dived by mean, since the signal is the relative change.
sig = sig / np.mean(sig, axis=1)[:, None]



fig, ax = plt.subplots()
ax.pcolormesh(t1, wl, sig, shading="gouraud")

# %%
# Since this is a little bit noisy, lets do the same for the average of all scans.
# First we average over the scans and then calculate the signal.

with h5.File(fpath, "r") as f:
    frames = f["frames/Probe1"]  # The data itself from the first probe.
    n_scans = len(frames)

    # Lets look at the data of the fith waiting time and the first scan.
    # This should correspond to 0.4 ps.
    data_mean = np.mean([frames["%d" % i]["2"][:] for i in range(n_scans)], axis=0)

    # If optical resolution is less than 1 pixel, we can smooth the data Along
    # the wavelength data to reduce noise. Here we use the savgol filter.
    from scipy.signal import savgol_filter
    data_mean = savgol_filter(data_mean, 11, 5, axis=0)

diff1 = (data_mean[:, ::4] - data_mean[:, 1::4])
diff2 = (data_mean[:, 2::4] - data_mean[:, 3::4])
diffm  = data_mean.mean(1)

sig_mean = np.log1p(diff1  / diffm[:, None]) + np.log1p(diff2 / diffm[:, None])

fig, ax = plt.subplots()
ax.pcolormesh(t1, wl, sig_mean, shading="gouraud", cmap="RdBu_r")

fig, (ax, ax2) = plt.subplots(2, figsize=(4, 3), constrained_layout=True)
n = 58
ax.plot(sig_mean[n], label="Channel %d, %d nm" % (n, wl[n]))
ax.legend()
ax2.plot(wl, sig_mean[:, 0], label="Channel 1, t1 = %.1f fs" % t1[0])
ax2.plot(wl, sig_mean[:, 1], label="Channel 1, t1 = %.1f fs" % t1[1])
ax2.plot(wl, sig_mean[:, 2], label="Channel 1, t1 = %.1f fs" % t1[2])
ax2.legend()

# %%
# Now we a have a nice inferogram. The next step is to do a Fourier transform.
# We will use the numpy fft module for this.
# The fourier transform is done along the t1 axis, since this is the axis that
# is modulated by the pump pulse.

zero_pad = 2
sig_mean[:, 0] *= 0.5
sig_ft = np.fft.fftshift(np.fft.fft(sig_mean, axis=1, n=zero_pad * sig_mean.shape[1]),
                         axes=1)

# The frequency axis is given by the inverse of the time axis.
# The zero frequency is in the middle of the axis.

freq = np.fft.fftshift(np.fft.fftfreq(sig_ft.shape[1], np.diff(t1)[0]))

# We want to plot the frequency in cm^-1, so we convert the frequency axis.
# Currently the frequency is in rad/ps, so we convert it to cm-1.

freq_cm = freq * 2 * np.pi * 33.35641 # 1 ps = 33.35641 cm^-1

fig, ax = plt.subplots()
from matplotlib.colors import TwoSlopeNorm
ax.pcolormesh(freq_cm, wl, sig_ft.real, shading="gouraud", cmap="RdBu_r", norm=TwoSlopeNorm(0))
ax.set_xlabel("Frequency [cm^-1]")
# %%
