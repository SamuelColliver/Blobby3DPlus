#!/usr/bin/env python3
"""Postprocess for the example galaxy.

Enhanced version with better emission line detection and multi-window support.

@author: Mathew Varidel
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import dnest4 as dn4
from pyblobby3d import PostBlobby3D
from pyblobby3d import SpectralModel

#path to the directory containing the data and the output
dir_path = '/Users/scol0322/Library/Mobile Documents/com~apple~CloudDocs/PhD-Code/Blobby3DPlus/examples/485885-windowed/'

# save current directory
original_dir = os.getcwd()

try:
    # change to target directory
    os.chdir(dir_path)
    
    # run DNest4 postprocessing
    # DNest4 assumes the level.txt file is in the current directory
    dn4.postprocess()
    
finally:
    # always return to original directory
    os.chdir(original_dir)

# read options file to get lines and nlines
def read_options_file(options_file):
    """read the options file to extract emission lines and nlines."""
    lines = []
    lsf_fwhm = None
    
    with open(options_file, 'r') as f:
        for line in f:
            # skip comments and empty lines
            if line.strip().startswith('#') or not line.strip():
                continue
            
            # remove inline comments
            if '#' in line:
                line = line[:line.index('#')]
            
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0].upper() == 'LINE':
                # parse line definition: LINE main_wavelength [coupled_wavelength1 ratio1 ...]
                line_data = []
                for i, part in enumerate(parts[1:]):
                    try:
                        line_data.append(float(part))
                    except ValueError:
                        break
                
                if line_data:
                    lines.append(line_data)
            
            elif parts[0].upper() == 'LSFFWHM':
                try:
                    lsf_fwhm = float(parts[1])
                except (ValueError, IndexError):
                    pass
    
    nlines = len(lines)
    return lines, nlines, lsf_fwhm

# read metadata file to get resolution and calculate lsf_fwhm for each window
def read_metadata_file(metadata_file):
    wavelength_windows = []
    
    with open(metadata_file, 'r') as f:
        for line in f:
            # skip comments and empty lines
            if line.strip().startswith('#') or not line.strip():
                continue
            
            # remove inline comments
            if '#' in line:
                line = line[:line.index('#')]
            
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0].upper() == 'WAVE_RANGE' and len(parts) >= 6:
                r_min = float(parts[1])
                r_max = float(parts[2])
                try:
                    # first try 7th column (index 6)
                    if len(parts) > 6:
                        lsf_fwhm = float(parts[6])
                    else:
                        lsf_fwhm = 1.5
                except (ValueError, IndexError):
                    # fallback calculation
                    central_wavelength = (r_min + r_max) / 2.0
                    lsf_fwhm = 1.5

                
                wavelength_windows.append({
                    'r_min': r_min,
                    'r_max': r_max,
                    'lsf_fwhm': lsf_fwhm
                })
    
    return wavelength_windows

def create_spectral_model_with_validation(emission_lines, lsf_fwhm_map, metadata):
    """create SpectralModel with validation and error handling."""
    try:
        # create the spectral model
        sm = SpectralModel(
            lines=emission_lines,
            lsf_fwhm=lsf_fwhm_map,
            wavelength_windows=metadata.wavelength_windows
        )
        
        # print summary
        sm.print_line_summary()
        
        # validate that lines are in windows
        validation = sm.validate_lines_in_windows()
        if not validation['all_valid']:
            print("\n!!!!! Warning: Some issues found with line-window compatibility:")
            for warning in validation['warnings']:
                print(f"  {warning}")
        
        return sm
        
    except Exception as e:
        print(f"!!!!!! ERROR creating SpectralModel: {e}")
        print("Falling back to default configuration...")
        
        # fallback to simple configuration
        fallback_lines = [[6562.81], [6583.1, 6548.1, 0.3333]][:len(emission_lines)]
        sm = SpectralModel(
            lines=fallback_lines,
            lsf_fwhm=1.61  # use global default
        )
        return sm

# read emission lines and metadata from files
options_file = os.path.join(dir_path, 'MODEL_OPTIONS')
emission_lines, nlines, lsf_fwhm_global = read_options_file(options_file)
print(f"Number of emission lines detected: {nlines}")
print("Emission lines:")
for i, line in enumerate(emission_lines):
    if len(line) == 1:
        print(f"  Line {i+1}: {line[0]} Å (single line)")
    else:
        main_line = line[0]
        coupled_info = []
        for j in range(1, len(line), 2):
            if j+1 < len(line):
                coupled_wave = line[j]
                ratio = line[j+1]
                coupled_info.append(f"{coupled_wave} Å (ratio: {ratio})")
        print(f"  Line {i+1}: {main_line} Å + coupled lines: {', '.join(coupled_info)}")

# read wavelength windows from metadata file
metadata_file = os.path.join(dir_path, 'metadata.txt')
wavelength_windows = read_metadata_file(metadata_file)
print(f"Number of wavelength windows: {len(wavelength_windows)}")

# create dictionary mapping window index to lsf_fwhm for multi-window support
lsf_fwhm_map = {}
for i, window in enumerate(wavelength_windows):
    lsf_fwhm_map[i] = window['lsf_fwhm']
    print(f"Window {i+1}: λ={window['r_min']:.1f}-{window['r_max']:.1f} Å, "
          f"LSF_FWHM={window['lsf_fwhm']:.3f} Å")

# now create PostBlobby3D object with full paths
post_b3d = PostBlobby3D(
        samples_path=dir_path+'posterior_sample.txt',
        data_path=dir_path+'data.txt',
        var_path=dir_path+'var.txt',
        metadata_path=dir_path+'metadata.txt',
        nlines=nlines)  # use detected number of lines

# choose a sample
sample = 0

# plot maps for sample - now supporting multiple lines
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

# plot maps for all detected lines
for line_idx in range(min(nlines, 2)):  # limit to first 2 lines for display
    if line_idx == 0:
        # first line - typically H-alpha
        main_wave = emission_lines[0][0] if emission_lines else 6562.81
        if abs(main_wave - 6562.81) < 0.1:
            title = r'H$\alpha$ Flux'
        else:
            title = f'Line 1 ({main_wave:.1f} Å) Flux'
    elif line_idx == 1:
        # second line - typically [NII]
        main_wave = emission_lines[1][0] if len(emission_lines) > 1 else 6583.1
        if abs(main_wave - 6583.1) < 0.1:
            title = r'[NII] Flux'
        else:
            title = f'Line 2 ({main_wave:.1f} Å) Flux'
    
    ax[line_idx].set_title(title)
    flux_map = post_b3d.maps[sample, line_idx]
    # handle zero or negative fluxes in log plot
    flux_map_safe = np.where(flux_map > 0, flux_map, np.nan)
    im = ax[line_idx].imshow(
        np.log10(flux_map_safe),
        interpolation='nearest', origin='lower')
    plt.colorbar(im, ax=ax[line_idx], label='log10(Flux)')

# if only one line detected, show placeholder for second
if nlines < 2:
    ax[1].set_title('No Second Line')
    ax[1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax[1].transAxes)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

ax[2].set_title('V (km/s)')
im2 = ax[2].imshow(post_b3d.maps[sample, nlines], interpolation='nearest', origin='lower')
plt.colorbar(im2, ax=ax[2], label='km/s')

ax[3].set_title('V Disp (km/s)')
im3 = ax[3].imshow(post_b3d.maps[sample, nlines+1], interpolation='nearest', origin='lower')
plt.colorbar(im3, ax=ax[3], label='km/s')

fig.tight_layout()
plt.show()

# we can also plot the integrated flux across the wavelength axis for a sample
# and compare it to the data
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

ax[0].set_title('Preconvolved')
preconv_sum = post_b3d.precon_cubes[sample].sum(axis=2)
preconv_sum_safe = np.where(preconv_sum > 0, preconv_sum, np.nan)
im0 = ax[0].imshow(np.log10(preconv_sum_safe), interpolation='nearest', origin='lower')
plt.colorbar(im0, ax=ax[0])

ax[1].set_title('Convolved')
conv_sum = post_b3d.con_cubes[sample].sum(axis=2)
conv_sum_safe = np.where(conv_sum > 0, conv_sum, np.nan)
im1 = ax[1].imshow(np.log10(conv_sum_safe), interpolation='nearest', origin='lower')
plt.colorbar(im1, ax=ax[1])

ax[2].set_title('Data')
data_sum = post_b3d.data.sum(axis=2)
data_sum_safe = np.where(data_sum > 0, data_sum, np.nan)
im2 = ax[2].imshow(np.log10(data_sum_safe), interpolation='nearest', origin='lower')
plt.colorbar(im2, ax=ax[2])

ax[3].set_title('Residuals (data - convolved)')
residual_map = (np.log10(data_sum_safe) - np.log10(conv_sum_safe))
im3 = ax[3].imshow(residual_map, interpolation='nearest', origin='lower', cmap='RdBu_r')
# add colorbar to make sure the centre is white
cbar = plt.colorbar(im3, ax=ax[3])
cbar.set_label('Residual (log scale)')

fig.tight_layout()
plt.show()

# plot showing the residuals at each central line location for all windows
if len(wavelength_windows) > 0:
    fig, axes = plt.subplots(1, len(wavelength_windows), figsize=(5*len(wavelength_windows), 4))
    if len(wavelength_windows) == 1:
        axes = [axes]

    central_i, central_j = post_b3d.data.shape[0]//2, post_b3d.data.shape[1]//2

    for w, window in enumerate(wavelength_windows):
        # find the wavelength indices for this window more carefully
        r_full = post_b3d.metadata.r_full
        
        # find indices within this window's range
        window_mask = (r_full >= window['r_min']) & (r_full <= window['r_max'])
        window_indices = np.where(window_mask)[0]
        
        if len(window_indices) == 0:
            # no data in this window
            axes[w].text(0.5, 0.5, 'No data\nin window', ha='center', va='center', 
                        transform=axes[w].transAxes)
            axes[w].set_title(f'Window {w+1}: {window["r_min"]:.0f}-{window["r_max"]:.0f} Å')
            continue
        
        wavelengths = r_full[window_indices]
        data_spectrum = post_b3d.data[central_i, central_j, window_indices]
        model_spectrum = post_b3d.con_cubes[sample, central_i, central_j, window_indices]
        residuals = data_spectrum - model_spectrum
        
        axes[w].plot(wavelengths, data_spectrum, 'k-', label='Data', alpha=0.7, linewidth=1.5)
        axes[w].plot(wavelengths, model_spectrum, 'r-', label='Model', linewidth=1.5)
        axes[w].plot(wavelengths, residuals, 'b-', label='Residuals', alpha=0.7)
        
        axes[w].set_xlabel('Wavelength (Å)')
        axes[w].set_ylabel('Flux')
        axes[w].set_title(f'Window {w+1}: {window["r_min"]:.0f}-{window["r_max"]:.0f} Å')
        axes[w].legend()
        axes[w].grid(True, alpha=0.3)

    fig.suptitle(f'Spectral fit at central spaxel ({central_i}, {central_j})')
    fig.tight_layout()
    plt.show()

# similarly we can integrate the total cube and look at the flux as
# a function of wavelength
fig, ax = plt.subplots(figsize=(12, 6))

# plot data
ax.plot(post_b3d.metadata.r_full, post_b3d.data.sum(axis=(0, 1)), 
        '--k', label='Data', linewidth=2)

# plot model samples
n_samples_to_plot = min(5, post_b3d.con_cubes.shape[0])
for s in range(n_samples_to_plot):
    alpha = 0.3 if s > 0 else 0.8
    label = 'Models' if s == 0 else None
    ax.plot(post_b3d.metadata.r_full, post_b3d.con_cubes[s].sum(axis=(0, 1)), 
            color='red', alpha=alpha, label=label)

ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel('Integrated Flux')
ax.set_title('Integrated flux across all spaxels')
ax.legend()
ax.grid(True, alpha=0.3)

# highlight wavelength windows
y_min, y_max = ax.get_ylim()
for w, window in enumerate(wavelength_windows):
    color = f'C{w % 10}'  # cycle through colors
    ax.axvspan(window['r_min'], window['r_max'], alpha=0.1, 
               color=color, label=f'Window {w+1}' if w < 5 else None)

# add vertical lines for emission lines
for i, line in enumerate(emission_lines):
    main_wave = line[0]
    ax.axvline(main_wave, color=f'C{i % 10}', linestyle=':', alpha=0.7, 
               label=f'Line {i+1}: {main_wave:.1f} Å' if i < 3 else None)
    
    # add coupled lines
    if len(line) > 1:
        for j in range(1, len(line), 2):
            if j+1 < len(line):
                coupled_wave = line[j]
                ax.axvline(coupled_wave, color=f'C{i % 10}', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# create SpectralModel with validation
sm = create_spectral_model_with_validation(emission_lines, lsf_fwhm_map, post_b3d.metadata)

# fit the data cube
wave = post_b3d.metadata.get_axis_array('r')
fit, fit_err = sm.fit_cube(wave, post_b3d.data, post_b3d.var)

# velocity dispersion comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Velocity Dispersion Comparison')

vdisp_map_idx = nlines + 1  # velocity dispersion is after flux and velocity maps

# preconvolved velocity dispersion
vmin, vmax = 5.0, 80.0
im1 = ax[0].imshow(post_b3d.maps[sample, vdisp_map_idx], vmin=vmin, vmax=vmax, 
                   origin='lower', cmap='viridis')
ax[0].set_title('Preconvolved V Disp (km/s)')
ax[0].set_xlabel('Pixel X')
ax[0].set_ylabel('Pixel Y')
plt.colorbar(im1, ax=ax[0], label='km/s')

# fitted velocity dispersion from convolved data
if fit.shape[0] > 3:  # ensure we have velocity dispersion in fit results
    im2 = ax[1].imshow(fit[3], vmin=vmin, vmax=vmax, origin='lower', cmap='viridis')
    ax[1].set_title('Fitted V Disp (km/s)')
    ax[1].set_xlabel('Pixel X')
    ax[1].set_ylabel('Pixel Y')
    plt.colorbar(im2, ax=ax[1], label='km/s')
else:
    ax[1].text(0.5, 0.5, 'Fit failed\nor insufficient\nparameters', 
               ha='center', va='center', transform=ax[1].transAxes)
    ax[1].set_title('Fitted V Disp (Failed)')

fig.tight_layout()
plt.show()

# plot showing the residuals of the v and v_disp maps
if fit.shape[0] > 3:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # velocity residuals
    v_residual = post_b3d.maps[sample, nlines] - fit[2]  # velocity map index
    im1 = axes[0].imshow(v_residual, origin='lower', cmap='RdBu_r')
    axes[0].set_title('Velocity Residuals (Preconv - Fitted)')
    axes[0].set_xlabel('Pixel X')
    axes[0].set_ylabel('Pixel Y')
    plt.colorbar(im1, ax=axes[0], label='km/s')

    # velocity dispersion residuals
    vdisp_residual = post_b3d.maps[sample, nlines+1] - fit[3]
    im2 = axes[1].imshow(vdisp_residual, origin='lower', cmap='RdBu_r')
    axes[1].set_title('V Disp Residuals (Preconv - Fitted)')
    axes[1].set_xlabel('Pixel X')
    axes[1].set_ylabel('Pixel Y')
    plt.colorbar(im2, ax=axes[1], label='km/s')

    fig.suptitle('Kinematic Residuals: Preconvolved - Fitted')
    fig.tight_layout()
    plt.show()

print(f"Processed {nlines} emission lines across {len(wavelength_windows)} wavelength windows")
print("Detected emission lines from MODEL_OPTIONS:")
for i, line in enumerate(emission_lines):
    line_info = f"  Line {i+1}: {line[0]:.2f} Å"
    if len(line) > 1:
        coupled = []
        for j in range(1, len(line), 2):
            if j+1 < len(line):
                coupled.append(f"{line[j]:.2f} Å (ratio: {line[j+1]:.3f})")
        if coupled:
            line_info += f" + {', '.join(coupled)}"
    print(line_info)