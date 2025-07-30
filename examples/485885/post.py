#!/usr/bin/env python3
"""Postprocess for the example galaxy.

The PostBlobby3D class is very simple. However, it is useful for organisational
purposes of the Blobby3D output. In this script, I created a PostBlobby3D
object and plotted a handful of sample attributes.

@author: Mathew Varidel

"""

import numpy as np
import matplotlib.pyplot as plt

from dnest4 import postprocess
from pyblobby3d import PostBlobby3D
from pyblobby3d import SpectralModel

postprocess()

post_b3d = PostBlobby3D(
        samples_path='posterior_sample.txt',
        data_path='data.txt',
        var_path='var.txt',
        metadata_path='metadata.txt',
        nlines=2)

# choose a sample
sample = 0

# Plot maps for sample
fig, ax = plt.subplots(1, 4)

ax[0].set_title(r'H$\alpha$ Flux')
ax[0].imshow(
    np.log10(post_b3d.maps[sample, 0]),
    interpolation='nearest', origin='lower')

ax[1].set_title(r'[NII] Flux')
ax[1].imshow(
    np.log10(post_b3d.maps[sample, 1]),
    interpolation='nearest', origin='lower')

ax[2].set_title('V')
ax[2].imshow(post_b3d.maps[sample, 2], interpolation='nearest', origin='lower')

ax[3].set_title('V Disp')
ax[3].imshow(post_b3d.maps[sample, 3], interpolation='nearest', origin='lower')

fig.tight_layout()

# We can also plot the integrated flux across the wavelength axis for a sample
# and compare it to the data. The below does this for H-alpha.
fig, ax = plt.subplots(1, 3)

ax[0].set_title('Preconvolved')
ax[0].imshow(
        np.log10(post_b3d.precon_cubes[sample].sum(axis=2)),
        interpolation='nearest', origin='lower')

ax[1].set_title('Convolved')
ax[1].imshow(
        np.log10(post_b3d.con_cubes[sample].sum(axis=2)),
        interpolation='nearest', origin='lower')

ax[2].set_title('Data')
ax[2].imshow(
        np.log10(post_b3d.data.sum(axis=2)),
        interpolation='nearest', origin='lower')

fig.tight_layout()

# Similarly we can integrate the total cube and look at the flux as
# a function of wavelength. In this case lets compare all convolved samples
# to the data.
fig, ax = plt.subplots()

ax.plot(post_b3d.data.sum(axis=(0, 1)), '--k', label='Data')
ax.plot(post_b3d.con_cubes.sum(axis=(1, 2)).transpose(), '--', color='0.5')

ax.legend()

# Another interesting thing can be to compare the velocity dispersion of the
# models pre and post convolution. The post emission lines in the convolved
# are not known analytically, and thus need to be estimated. There is an
# emission line fitting procedure for this purpose in pyblobby3d. You will
# often see that a flat velocity dispersion leads to a velocity dispersion map
# with substructure.
sm = SpectralModel(
        lines=[[6562.81], [6583.1, 6548.1, 0.3333]],
        lsf_fwhm=1.61,)

wave = post_b3d.metadata.get_axis_array('r')
fit, fit_err = sm.fit_cube(wave, post_b3d.data, post_b3d.var)
fit_model = sm.calculate_cube(wave, fit)

fig, ax = plt.subplots(1, 2)
fig.suptitle('V Disp')

ax[0].set_title('Preconvolved')
ax[0].imshow(post_b3d.maps[sample, 3], vmin=10.0, vmax=50.0)

ax[1].set_title('Convolved')
ax[1].imshow(fit[3], vmin=10.0, vmax=50.0)

fig.tight_layout()

plt.show()