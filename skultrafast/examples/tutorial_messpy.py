"""
Working with MessPy 1 files
---------------------------

This example shows how to load files from MessPy v1, hence it is only of interested for
people working with files produced by it.

MessPy v1 files are .npz files, which consists of zipped npy (numpy) files. Under
the module messpy we a helper class to work with it. We will start with importing
the module and the standard toools.
"""
# %%
from skultrafast import messpy, dataset
import matplotlib.pyplot as plt
import skultrafast
print(skultrafast.__version__)

plt.rcParams['figure.dpi'] = 130
plt.rcParams['figure.size'] = (3.2, 2)

# %%
# The main tool is the `MessPyFile` class. Note the constructor takes all the neccesary
# information to do the processing. Here I will pass all parameters explictily for
# documentation proposes. Some of the parameters can be infered automatically.
path = r'Z:\Datatemp\2018\ir\2018-11-16 Chlorophyll A'
fname = path + r"\till_chlA_10ps6.npz"
mpf = messpy.MessPyFile(
    fname,
    invert_data=True,  # Changes the sign of the data
    is_pol_resolved=True,  # If the data was recored polarization resolved.
    pol_first_scan='perp',  # Polarisation of the first scan
    valid_channel=1,  # Which channel to use, recent IR data always uses 1
    # Recent visible data uses 0
)

print(mpf.data.shape)
# %%
# Simlar to TimeResSpec the MessPyFile class has a plotter subclass with various
# plot methods. For example, the `compare_spec` method plots a averaged spectrum for
# each central channel recored.

mpf.plot.compare_spec()

# %%
# As we can see, the applied wavelength calibration used by messpy was not correct.
# Let's change that.

mpf.recalculate_wavelengths(8.8)
mpf.plot.compare_spec()

# %%
# Note that `MessPyFile` uses sigma clipping when averaging the scans. If you
# want more control over the process, use the `average_scans` method. For example
# here we change the clipping range and use only the first 5 scans.

mpf.average_scans(sigma=2, max_scan=5)
# %%
# The indivudal datasets, for each polarization and each spectral window can be
# found in a dict belonging to the class.

for key, ds in mpf.av_scans_.items():
    print(key, ds)

# %%
# Now we can work with them directly. For example datasets can be combined manually

iso_merge = mpf.av_scans_['iso0'].concat_datasets(mpf.av_scans_['iso1'])
all_iso = iso_merge.concat_datasets(mpf.av_scans_['iso2'])

# %%
# Since this is quite common, we can also use the avg_and_concat methods,
# which automates this.

para, perp, iso = mpf.avg_and_concat()
iso.plot.spec(1, 3, 10, n_average=5)

# %%
# Why does the spectrum now look so yanky? Because now neighboring points
# where recorded indifferent spectral windows and hence their noise differ, while the
# noise within one window is often correlated. Also the spectrum now suggest are larger
# spectral resolution the it has. Hence, the mitigate both points we can bin down
# our spectrum. Either we uniformally bin or only merge channgels which are too close
# together.

fig, (ax0, ax1) = plt.subplots(2, figsize=(3,4), sharex=True)

bin_iso = iso.bin_freqs(30)
bin_iso.plot.spec(1, 3, 10, n_average=5, marker='o', ax=ax0, ms=3)

merge_iso = iso.merge_nearby_channels(8)
merge_iso.plot.spec(1, 3, 10, n_average=5, marker='o', ax=ax1, ms=3)

# Remove Legend and correct ylabel
ax0.legend_ = None
ax0.yaxis.label.set_position((0, 0.0))
ax1.legend_ = None
ax1.set_ylabel('')

# %%
# The work with are polarisation resolved transient spectrum we use `PolTRSpec`, which
# takes the two datasets we get from avg_and_concat.

pol_ds = dataset.PolTRSpec(para, perp)
merged_ds = pol_ds.merge_nearby_channels(8)
merged_ds.plot.spec(1, n_average=4)

