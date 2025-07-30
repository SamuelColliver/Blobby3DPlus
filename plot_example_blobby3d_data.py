#!/usr/bin/env python3
"""
Simple script that visualises the example Blobby3D data 
-> almost acts as a dry run to the trim code which will create a new dir for trimmed data

TODO::
-> remove the other plotting option
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_metadata(metadata_path='metadata.txt'):
    """Load metadata and parse cube dimensions with new format."""
    
    # initialise variables
    metadata = {}
    wavelength_ranges = []
    
    with open(metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # remove inline comments
            if '#' in line:
                line = line[:line.index('#')].strip()
            
            parts = line.split()
            if len(parts) < 2:
                continue
                
            keyword = parts[0].lower()
            
            if keyword == 'ni':
                metadata['ni'] = int(parts[1])
            elif keyword == 'nj':
                metadata['nj'] = int(parts[1])
            elif keyword == 'x_min':
                metadata['x_min'] = float(parts[1])
            elif keyword == 'x_max':
                metadata['x_max'] = float(parts[1])
            elif keyword == 'y_min':
                metadata['y_min'] = float(parts[1])
            elif keyword == 'y_max':
                metadata['y_max'] = float(parts[1])
            elif keyword == 'wave_range':
                if len(parts) >= 6:
                    wave_range = {
                        'r_min': float(parts[1]),
                        'r_max': float(parts[2]),
                        'start_col': int(parts[3]),
                        'end_col': int(parts[4]),
                        'n_bins': int(parts[5])
                    }
                    wavelength_ranges.append(wave_range)
    
    # validate required parameters
    required_keys = ['ni', 'nj', 'x_min', 'x_max', 'y_min', 'y_max']
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Required parameter '{key}' not found in metadata file")
    
    if not wavelength_ranges:
        raise ValueError("No wavelength ranges found in metadata file")
    
    # calculate dimensions
    naxis = [metadata['ni'], metadata['nj'], sum(wr['n_bins'] for wr in wavelength_ranges)]
    x_lim = [metadata['x_min'], metadata['x_max']]
    y_lim = [metadata['y_min'], metadata['y_max']]
    r_lim = [wavelength_ranges[0]['r_min'], wavelength_ranges[-1]['r_max']]
    
    # calculate pixel sizes
    dx = (x_lim[1] - x_lim[0]) / naxis[1]
    dy = (y_lim[1] - y_lim[0]) / naxis[0]
    
    # create coordinate arrays
    x = np.linspace(x_lim[0] + 0.5*dx, x_lim[1] - 0.5*dx, naxis[1])
    y = np.linspace(y_lim[0] + 0.5*dy, y_lim[1] - 0.5*dy, naxis[0])
    
    # create combined wavelength array
    wave = []
    for wr in wavelength_ranges:
        dr = (wr['r_max'] - wr['r_min']) / wr['n_bins']
        wave_range = np.linspace(wr['r_min'] + 0.5*dr, wr['r_max'] - 0.5*dr, wr['n_bins'])
        wave.extend(wave_range)
    
    wave = np.array(wave)
    
    # calculate average dr for compatibility
    total_range = r_lim[1] - r_lim[0]
    dr = total_range / naxis[2]
    
    return {
        'naxis': naxis,
        'x_lim': x_lim, 'y_lim': y_lim, 'r_lim': r_lim,
        'dx': dx, 'dy': dy, 'dr': dr,
        'x': x, 'y': y, 'wave': wave,
        'wavelength_ranges': wavelength_ranges
    }

def load_data(data_path, metadata):
    """Load data cube from text file."""
    data_flat = np.loadtxt(data_path)
    # reshape from flat array to 3D cube (ni, nj, nr)
    data_cube = data_flat.reshape(metadata['naxis'])
    return data_cube

def add_wavelength_windows(ax, expected_lines, wave_range, window_size=35):
    """Add wavelength windows (±window_size Å) around emission lines."""
    colors = {'H-alpha': 'cornflowerblue', '[NII]6583': 'forestgreen', '[NII]6548': 'indianred',
              '[OII]3727': 'gold', 'H-beta': 'hotpink', '[OIII]5007': 'dimgray',
              '[SII]6717': 'steelblue', '[SII]6731': 'steelblue'}
    
    for name, line_wave in expected_lines.items():
        if wave_range[0] <= line_wave <= wave_range[1]:
            # window around line
            ax.axvspan(line_wave - window_size, line_wave + window_size, 
                      alpha=0.25, color=colors.get(name, 'lightgray'), 
                      label=f'±{window_size}Å ({name})' if name == 'H-alpha' else "")
            
            # line center
            ax.axvline(line_wave, color='red', linestyle='--', alpha=0.8, linewidth=1)

def plot_data_overview(data_cube, var_cube, metadata, all_lines, window_size=35):
    """Create overview plots of the data cube."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # filter lines that are in the current wavelength range
    expected_lines = {name: wave for name, wave in all_lines.items() 
                     if metadata['r_lim'][0] <= wave <= metadata['r_lim'][1]}
    
    # 1. Total flux map (sum over wavelength)
    flux_total = np.sum(data_cube, axis=2)
    im1 = axes[0,0].imshow(flux_total, origin='lower', aspect='auto', 
                          extent=[metadata['x_lim'][0], metadata['x_lim'][1],
                                 metadata['y_lim'][0], metadata['y_lim'][1]])
    axes[0,0].set_title('Total Flux Map\n(Sum over wavelength)')
    axes[0,0].set_xlabel('X (arcsec)')
    axes[0,0].set_ylabel('Y (arcsec)')
    plt.colorbar(im1, ax=axes[0,0])
    
    # 2. Central spectrum (middle spaxel) with windows
    center_i, center_j = metadata['naxis'][0]//2, metadata['naxis'][1]//2
    spectrum = data_cube[center_i, center_j, :]
    
    # Add wavelength windows first (so they appear behind the spectrum)
    add_wavelength_windows(axes[0,1], expected_lines, metadata['r_lim'], window_size)
    
    axes[0,1].plot(metadata['wave'], spectrum, 'b-', linewidth=1.5, label='Spectrum')
    axes[0,1].set_title(f'Central Spaxel Spectrum (±{window_size}Å windows)\n(i={center_i}, j={center_j})')
    axes[0,1].set_xlabel('Wavelength (Å)')
    axes[0,1].set_ylabel('Flux')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add legend for windows (only show a few to avoid clutter)
    handles, labels = axes[0,1].get_legend_handles_labels()
    # Keep spectrum line and first few window entries
    if len(handles) > 3:
        axes[0,1].legend(handles[:3], labels[:3], loc='upper right', fontsize=8)
    else:
        axes[0,1].legend(loc='upper right', fontsize=8)
    
    # 3. Signal-to-noise map
    with np.errstate(divide='ignore', invalid='ignore'):
        snr_map = np.sum(data_cube, axis=2) / np.sqrt(np.sum(var_cube, axis=2))
        snr_map[~np.isfinite(snr_map)] = 0
    
    im3 = axes[0,2].imshow(snr_map, origin='lower', aspect='auto',
                          extent=[metadata['x_lim'][0], metadata['x_lim'][1],
                                 metadata['y_lim'][0], metadata['y_lim'][1]])
    axes[0,2].set_title('Signal-to-Noise Map')
    axes[0,2].set_xlabel('X (arcsec)')
    axes[0,2].set_ylabel('Y (arcsec)')
    plt.colorbar(im3, ax=axes[0,2])
    
    # 4. Wavelength slice (middle wavelength)
    mid_wave_idx = metadata['naxis'][2] // 2
    wave_slice_start = max(0, mid_wave_idx - 10)
    wave_slice_end = min(metadata['naxis'][2], mid_wave_idx + 10)
    wave_slice = np.sum(data_cube[:, :, wave_slice_start:wave_slice_end], axis=2)
    
    im4 = axes[1,0].imshow(wave_slice, origin='lower', aspect='auto',
                          extent=[metadata['x_lim'][0], metadata['x_lim'][1],
                                 metadata['y_lim'][0], metadata['y_lim'][1]])
    axes[1,0].set_title(f'Wavelength Slice\n(∫ from λ = {metadata["wave"][wave_slice_start]:.1f}Å to {metadata["wave"][wave_slice_end-1]:.1f}Å)')
    axes[1,0].set_xlabel('X (arcsec)')
    axes[1,0].set_ylabel('Y (arcsec)')
    plt.colorbar(im4, ax=axes[1,0])
    
    # 5. Integrated spectrum (sum over spatial dimensions) with windows
    total_spectrum = np.sum(data_cube, axis=(0,1))
    
    # Add wavelength windows first
    add_wavelength_windows(axes[1,1], expected_lines, metadata['r_lim'], window_size)
    
    axes[1,1].plot(metadata['wave'], total_spectrum, 'g-', linewidth=1.5, label='Integrated Spectrum')
    axes[1,1].set_title(f'Integrated Spectrum with ±{window_size}Å Windows\n(Sum over all spaxels)')
    axes[1,1].set_xlabel('Wavelength (Å)')
    axes[1,1].set_ylabel('Total Flux')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add text annotations for window sizes and wavelength ranges
    range_text = "Wavelength ranges:\n"
    for i, wr in enumerate(metadata['wavelength_ranges']):
        range_text += f"  {wr['r_min']:.1f}-{wr['r_max']:.1f}Å ({wr['n_bins']} bins)\n"
    
    axes[1,1].text(0.02, 0.98, range_text.strip(), 
                   transform=axes[1,1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
    
    # 6. Variance overview
    var_total = np.sum(var_cube, axis=2)
    im6 = axes[1,2].imshow(var_total, origin='lower', aspect='auto',
                          extent=[metadata['x_lim'][0], metadata['x_lim'][1],
                                 metadata['y_lim'][0], metadata['y_lim'][1]])
    axes[1,2].set_title('Total Variance Map')
    axes[1,2].set_xlabel('X (arcsec)')
    axes[1,2].set_ylabel('Y (arcsec)')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.tight_layout()
    return fig

def print_data_summary(data_cube, var_cube, metadata):
    """Print summary statistics of the data."""
    print("=" * 60)
    print("BLOBBY3D DATA SUMMARY")
    print("=" * 60)
    print(f"Data cube shape: {data_cube.shape} (ni, nj, nr)")
    print(f"Spatial dimensions: {metadata['naxis'][0]} × {metadata['naxis'][1]} spaxels")
    print(f"Spectral dimension: {metadata['naxis'][2]} wavelength bins")
    print(f"Wavelength coverage: {metadata['r_lim'][0]:.2f} - {metadata['r_lim'][1]:.2f} Å")
    print(f"Average wavelength resolution: {metadata['dr']:.3f} Å per bin")
    print(f"Spatial coverage: {metadata['x_lim'][0]:.2f}″ to {metadata['x_lim'][1]:.2f}″ (X)")
    print(f"                  {metadata['y_lim'][0]:.2f}″ to {metadata['y_lim'][1]:.2f}″ (Y)")
    print(f"Pixel scale: {metadata['dx']:.3f}″ × {metadata['dy']:.3f}″")
    print()
    
    print("WAVELENGTH RANGES:")
    for i, wr in enumerate(metadata['wavelength_ranges']):
        print(f"  Range {i+1}: {wr['r_min']:.2f} - {wr['r_max']:.2f} Å ({wr['n_bins']} bins)")
        print(f"           Columns {wr['start_col']} - {wr['end_col']}")
    print()
    
    print("DATA STATISTICS:")
    print(f"  Total flux: {np.sum(data_cube):.2e}")
    print(f"  Max flux (single bin): {np.max(data_cube):.2e}")
    print(f"  Min flux (single bin): {np.min(data_cube):.2e}")
    print(f"  RMS noise: {np.sqrt(np.mean(var_cube)):.2e}")
    
    # count valid pixels (non-zero variance)
    valid_mask = var_cube > 0
    n_valid = np.sum(valid_mask)
    n_total = var_cube.size
    print(f"  Valid pixels: {n_valid} / {n_total} ({100*n_valid/n_total:.1f}%)")
    
    # find brightest spaxel
    flux_map = np.sum(data_cube, axis=2)
    max_i, max_j = np.unravel_index(np.argmax(flux_map), flux_map.shape)
    print(f"  Brightest spaxel: ({max_i}, {max_j}) with flux {flux_map[max_i, max_j]:.2e}")
    
    print("=" * 60)

def analyse_emission_lines(data_cube, metadata, expected_lines, window_size=35):
    """Analyse emission lines in the integrated spectrum."""
    print("\nEMISSION LINE ANALYSIS:")
    print("-" * 30)
    
    # integrated spectrum
    total_spectrum = np.sum(data_cube, axis=(0,1))
    wave = metadata['wave']
    
    
    # add info about wavelength windows
    print(f"\nWAVELENGTH WINDOW ANALYSIS (±{window_size}Å):")
    print("-" * 40)
    for name, line_wave in expected_lines.items():
        if metadata['r_lim'][0] <= line_wave <= metadata['r_lim'][1]:
            window_min, window_max = line_wave - window_size, line_wave + window_size
            
            print(f"  {name:12s}:")
            print(f"    ±{window_size}Å window: {window_min:.1f} - {window_max:.1f} Å")
            
            # calculate flux in window
            mask = (wave >= window_min) & (wave <= window_max)
            flux_window = np.sum(total_spectrum[mask]) if np.any(mask) else 0
            
            print(f"    Flux in ±{window_size}Å: {flux_window:.2e}")
            print()
    
    print("LINE ANALYSIS:")
    print("-" * 15)
    for name, line_wave in expected_lines.items():
        if metadata['r_lim'][0] <= line_wave <= metadata['r_lim'][1]:
            # find closest wavelength bin
            idx = np.argmin(np.abs(wave - line_wave))
            actual_wave = wave[idx]
            flux = total_spectrum[idx]
            
            # estimate continuum (average of bins ±10Å away)
            cont_mask = (np.abs(wave - line_wave) > 10) & (np.abs(wave - line_wave) < 20)
            if np.any(cont_mask):
                continuum = np.mean(total_spectrum[cont_mask])
            else:
                continuum = 0
                
            print(f"  {name:12s}: λ={actual_wave:7.2f}Å, flux={flux:.2e}, S/C={flux/max(continuum,1e-10):.1f}")
        else:
            print(f"  {name:12s}: Outside wavelength range")

def main():
    """Main function to load and visualise data."""
    
    # +++++++++++++ Can be Edited +++++++++++++++++

    WINDOW_SIZE = 5.0  # size (radius; Angstrom) of the window (ie. 8 Angstrom = 16 total)
    
    # file paths
    data_file = './examples/485885/data.txt'
    var_file = './examples/485885/var.txt'
    metadata_file = './examples/485885/metadata.txt'

    # define emission lines for different wavelength ranges
    all_lines = {
        'H-beta': 4861.3,
        '[OIII]5007': 5006.8,
        'H-alpha': 6562.81,
        '[NII]6548': 6548.1,
        '[NII]6583': 6583.1,
        '[SII]6717': 6716.4,
        '[SII]6731': 6730.8
    }

    # +++++++++ End of can be Edited +++++++++++++

    # check files exist
    for file in [data_file, var_file, metadata_file]:
        if not Path(file).exists():
            print(f"Error: {file} not found in current directory")
            return
    
    print("Loading Blobby3D data...")
    print(f"Using ±{WINDOW_SIZE}Å windows around emission lines")
    
    try:
        # load data
        metadata = load_metadata(metadata_file)
        data_cube = load_data(data_file, metadata)
        var_cube = load_data(var_file, metadata)
        
        # print summary
        print_data_summary(data_cube, var_cube, metadata)
        
        # analyse emission lines
        analyse_emission_lines(data_cube, metadata, all_lines, WINDOW_SIZE)
        
        # create plots
        print("\nCreating visualisation...")
        fig = plot_data_overview(data_cube, var_cube, metadata, all_lines, WINDOW_SIZE)
        
        # save plot
        output_filename = f'blobby3d_data_overview_window_{WINDOW_SIZE}A.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved as '{output_filename}'")
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == '__main__':
    main()