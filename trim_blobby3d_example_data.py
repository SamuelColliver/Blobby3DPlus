#!/usr/bin/env python3
"""
Blobby3D Multi-Window Data Trimmer (for the example data [or at least working Blobby3D data**])

** that has used the new metadata.txt format

This script processes Blobby3D data to extract multiple wavelength windows around
emission lines, combines overlapping windows, and updates metadata accordingly.

TODO::
-> try with new metadata.txt format if that doesn't work then try with old metadata format
    -> one wavelength bin and all one line
    In order: Ni, Nj, Nk, x_min, x_max, y_min, y_max, wave_min, wave_max.
    27 31 107 -7.500000 8.000000 -6.500000 7.000000 6535.076955 6595.057979
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_original_metadata(metadata_path):
    """Load original metadata from file with new format."""
    print(f"Loading original metadata from: {metadata_path}")
    
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
    
    # calculate overall wavelength coverage
    metadata['r_min'] = min(wr['r_min'] for wr in wavelength_ranges)
    metadata['r_max'] = max(wr['r_max'] for wr in wavelength_ranges)
    metadata['total_bins'] = sum(wr['n_bins'] for wr in wavelength_ranges)
    metadata['wavelength_ranges'] = wavelength_ranges
    
    # create combined wavelength array
    wavelengths = []
    for wr in wavelength_ranges:
        dr = (wr['r_max'] - wr['r_min']) / wr['n_bins']
        wave_range = np.linspace(wr['r_min'] + 0.5*dr, wr['r_max'] - 0.5*dr, wr['n_bins'])
        wavelengths.extend(wave_range)
    
    metadata['wavelengths'] = np.array(wavelengths)
    metadata['dr'] = (metadata['r_max'] - metadata['r_min']) / metadata['total_bins']  # Average
    
    print(f"  Original dimensions: {metadata['ni']} x {metadata['nj']} x {metadata['total_bins']}")
    print(f"  Wavelength coverage: {metadata['r_min']:.2f} - {metadata['r_max']:.2f} Å")
    print(f"  Number of ranges: {len(wavelength_ranges)}")
    for i, wr in enumerate(wavelength_ranges):
        print(f"    Range {i+1}: {wr['r_min']:.2f} - {wr['r_max']:.2f} Å ({wr['n_bins']} bins)")
    
    return metadata

def load_data_cube(data_path, metadata):
    """Load and reshape data cube."""
    print(f"Loading data cube from: {data_path}")
    data_flat = np.loadtxt(data_path)
    
    # reshape from flat array to 3D cube (ni, nj, nr)
    expected_size = metadata['ni'] * metadata['nj'] #* metadata['total_bins']
    if len(data_flat) != expected_size:
        raise ValueError(f"Data size mismatch: expected {expected_size}, got {len(data_flat)}")
    
    data_cube = data_flat.reshape(metadata['ni'], metadata['nj'], metadata['total_bins'])
    print(f"  Data cube shape: {data_cube.shape}")
    return data_cube

def define_emission_line_windows(emission_lines = {'H-alpha': 6562.81}, window_size=35):
    """Define wavelength windows around common emission lines."""
    

    
    # create windows around each line
    windows = []
    for name, center_wave in emission_lines.items():
        window = {
            'name': name,
            'center': center_wave,
            'r_min': center_wave - window_size,
            'r_max': center_wave + window_size,
            'width': 2 * window_size
        }
        windows.append(window)
    
    return windows

def filter_windows_by_coverage(windows, metadata):
    """Filter windows to only include those within the original wavelength range."""
    
    r_min_orig = metadata['r_min']
    r_max_orig = metadata['r_max']
    
    filtered_windows = []
    for window in windows:
        # check if window overlaps with original range
        if (window['r_max'] > r_min_orig and window['r_min'] < r_max_orig):
            # Clip window to original range
            clipped_window = window.copy()
            clipped_window['r_min'] = max(window['r_min'], r_min_orig)
            clipped_window['r_max'] = min(window['r_max'], r_max_orig)
            clipped_window['width'] = clipped_window['r_max'] - clipped_window['r_min']
            
            if clipped_window['width'] > 0:
                filtered_windows.append(clipped_window)
                print(f"  Including {window['name']}: {clipped_window['r_min']:.1f} - {clipped_window['r_max']:.1f} Å")
            else:
                print(f"  Excluding {window['name']}: outside wavelength range")
        else:
            print(f"  Excluding {window['name']}: outside wavelength range")
    
    return filtered_windows

def combine_overlapping_windows(windows, min_gap=5.0):
    """Combine overlapping or closely spaced windows."""
    
    if not windows:
        return []
    
    # sort windows by minimum wavelength
    sorted_windows = sorted(windows, key=lambda x: x['r_min'])
    
    combined = []
    current = sorted_windows[0].copy()
    current['lines'] = [current['name']]
    
    for window in sorted_windows[1:]:
        # check if windows overlap or are close (within min_gap)
        if window['r_min'] <= current['r_max'] + min_gap:
            # combine windows
            current['r_max'] = max(current['r_max'], window['r_max'])
            current['lines'].append(window['name'])
            current['name'] = ' + '.join(current['lines'])
            current['width'] = current['r_max'] - current['r_min']
        else:
            # start new window
            combined.append(current)
            current = window.copy()
            current['lines'] = [current['name']]
    
    # add the last window
    combined.append(current)
    
    print(f"\nCombined {len(windows)} windows into {len(combined)} non-overlapping windows:")
    for i, window in enumerate(combined):
        print(f"  Window {i+1}: {window['name']}")
        print(f"    Range: {window['r_min']:.1f} - {window['r_max']:.1f} Å ({window['width']:.1f} Å wide)")
        if 'lines' in window:
            print(f"    Lines: {', '.join(window['lines'])}")
    
    return combined

def extract_wavelength_indices(windows, metadata):
    """Convert wavelength ranges to bin indices for the combined wavelength array."""
    
    wavelengths = metadata['wavelengths']
    
    print(f"\nMapping windows to spectral bins:")
    for window in windows:
        # find closest indices in the combined wavelength array
        start_idx = np.argmin(np.abs(wavelengths - window['r_min']))
        end_idx = np.argmin(np.abs(wavelengths - window['r_max']))
        
        # ensure end_idx >= start_idx
        if end_idx < start_idx:
            end_idx = start_idx
        
        window['start_idx'] = start_idx
        window['end_idx'] = end_idx
        window['n_bins'] = end_idx - start_idx + 1
        
        # update actual wavelength range based on bin centers
        if start_idx < len(wavelengths) and end_idx < len(wavelengths):
            # calculate bin edges for accurate range
            if start_idx > 0:
                start_edge = (wavelengths[start_idx-1] + wavelengths[start_idx]) / 2
            else:
                start_edge = wavelengths[start_idx] - (wavelengths[1] - wavelengths[0]) / 2
            
            if end_idx < len(wavelengths) - 1:
                end_edge = (wavelengths[end_idx] + wavelengths[end_idx+1]) / 2
            else:
                end_edge = wavelengths[end_idx] + (wavelengths[-1] - wavelengths[-2]) / 2
            
            window['actual_r_min'] = start_edge
            window['actual_r_max'] = end_edge
        else:
            window['actual_r_min'] = window['r_min']
            window['actual_r_max'] = window['r_max']
        
        window['actual_width'] = window['actual_r_max'] - window['actual_r_min']
        
        print(f"  {window['name']}:")
        print(f"    Requested: {window['r_min']:.1f} - {window['r_max']:.1f} Å")
        print(f"    Actual: {window['actual_r_min']:.1f} - {window['actual_r_max']:.1f} Å")
        print(f"    Bins: {start_idx} - {end_idx} ({window['n_bins']} bins)")
    
    return windows

def extract_windowed_data(data_cube, var_cube, windows):
    """Extract data for the specified wavelength windows."""
    
    ni, nj, nr_orig = data_cube.shape
    
    # calculate total number of bins in windowed data
    total_bins = sum(window['n_bins'] for window in windows)
    
    # create new data cubes
    windowed_data = np.zeros((ni, nj, total_bins))
    windowed_var = np.zeros((ni, nj, total_bins))
    
    # extract data for each window
    current_bin = 0
    for window in windows:
        start_idx = window['start_idx']
        end_idx = window['end_idx']
        n_bins = window['n_bins']
        
        windowed_data[:, :, current_bin:current_bin+n_bins] = data_cube[:, :, start_idx:end_idx+1]
        windowed_var[:, :, current_bin:current_bin+n_bins] = var_cube[:, :, start_idx:end_idx+1]
        
        # update window with position in new array
        window['new_start_bin'] = current_bin
        window['new_end_bin'] = current_bin + n_bins - 1
        
        current_bin += n_bins
    
    print(f"\nExtracted windowed data:")
    print(f"  Original shape: {data_cube.shape}")
    print(f"  Windowed shape: {windowed_data.shape}")
    print(f"  Data reduction: {nr_orig} → {total_bins} bins ({100*total_bins/nr_orig:.1f}%)")
    
    return windowed_data, windowed_var

def write_windowed_data(windowed_data, windowed_var, output_dir):
    """Write windowed data to text files."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # flatten spatial via reshape
    data_flat = windowed_data.reshape(-1, windowed_data.shape[-1])
    var_flat = windowed_var.reshape(-1, windowed_var.shape[-1])
    
    data_file = output_dir / 'data.txt'
    var_file = output_dir / 'var.txt'
    
    print(f"\nWriting windowed data:")
    print(f"  Data: {data_file}")
    print(f"  Variance: {var_file}")
    
    np.savetxt(data_file, data_flat, fmt='%.8e')
    np.savetxt(var_file, var_flat, fmt='%.8e')
    
    return data_file, var_file

def write_metadata(metadata, windows, output_dir):
    """Write updated metadata file in the new format."""
    
    output_dir = Path(output_dir)
    metadata_file = output_dir / 'metadata.txt'
    
    print(f"  Metadata: {metadata_file}")
    
    with open(metadata_file, 'w') as f:
        f.write("# Blobby3D Metadata File - Multi-Window Configuration\n")
        f.write("# Generated by Blobby3D Multi-Window Data Trimmer\n\n")
        
        # spatial dimensions (unchanged)
        f.write(f"ni {metadata['ni']}\n")
        f.write(f"nj {metadata['nj']}\n")
        f.write(f"x_min {metadata['x_min']:.6f}\n")
        f.write(f"x_max {metadata['x_max']:.6f}\n")
        f.write(f"y_min {metadata['y_min']:.6f}\n")
        f.write(f"y_max {metadata['y_max']:.6f}\n")
        
        # wavelength windows in new format
        for i, window in enumerate(windows):
            f.write(f"wave_range {window['actual_r_min']:.6f} {window['actual_r_max']:.6f} "
                   f"{window['new_start_bin']} {window['new_end_bin']} {window['n_bins']}  "
                   f"# {window['name']}\n")
        
        f.write(f"\n# Total wavelength bins: {sum(w['n_bins'] for w in windows)}\n")
        f.write(f"# Original coverage: {metadata['r_min']:.2f} - {metadata['r_max']:.2f} Å\n")
        f.write(f"# Windowed coverage: {windows[0]['actual_r_min']:.2f} - {windows[-1]['actual_r_max']:.2f} Å\n")
    
    return metadata_file

def create_comparison_plot(original_data, windowed_data, metadata, windows, output_dir):
    """Create comparison plots showing original vs windowed data."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # original wavelengths
    orig_wave = metadata['wavelengths']
    wave_max = np.max(orig_wave)
    wave_min = np.min(orig_wave)

    
    # windowed wavelengths
    windowed_wave = []
    for window in windows:
        # extract wavelengths for this window from original array
        start_idx = window['start_idx']
        end_idx = window['end_idx']
        window_wave = orig_wave[start_idx:end_idx+1]
        windowed_wave.extend(window_wave)
    windowed_wave = np.array(windowed_wave)
    
    # spatial coordinates
    x_extent = [metadata['x_min'], metadata['x_max']]
    y_extent = [metadata['y_min'], metadata['y_max']]
    
    # 1. Total flux maps
    orig_flux = np.sum(original_data, axis=2)
    windowed_flux = np.sum(windowed_data, axis=2)
    
    im1 = axes[0,0].imshow(orig_flux, origin='lower', aspect='auto', 
                          extent=x_extent + y_extent)
    axes[0,0].set_title('Original Total Flux Map')
    axes[0,0].set_xlabel('X (arcsec)')
    axes[0,0].set_ylabel('Y (arcsec)')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(windowed_flux, origin='lower', aspect='auto',
                          extent=x_extent + y_extent)
    axes[0,1].set_title('Windowed Total Flux Map')
    axes[0,1].set_xlabel('X (arcsec)')
    axes[0,1].set_ylabel('Y (arcsec)')
    plt.colorbar(im2, ax=axes[0,1])
    
    # 2. Central spectra
    center_i, center_j = metadata['ni']//2, metadata['nj']//2
    orig_spectrum = original_data[center_i, center_j, :]
    windowed_spectrum = windowed_data[center_i, center_j, :]
    
    axes[1,0].plot(orig_wave, orig_spectrum, 'b-', linewidth=1, label='Original')
    axes[1,0].scatter(orig_wave, orig_spectrum) 
    axes[1,0].set_xlim(wave_min,wave_max)
    axes[1,0].set_title(f'Original Central Spectrum\n(spaxel {center_i}, {center_j})')
    axes[1,0].set_xlabel('Wavelength (Å)')
    axes[1,0].set_ylabel('Flux')
    axes[1,0].grid(True, alpha=0.3)
    
    # add window regions to original spectrum
    colors = plt.cm.Set3(np.linspace(0, 1, len(windows)))
    for i, window in enumerate(windows):
        axes[1,0].axvspan(window['actual_r_min'], window['actual_r_max'], 
                         alpha=0.3, color=colors[i], label=window['name'])
    
    # only show legend if we have few windows to avoid clutter
    if len(windows) <= 5:
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    axes[1,1].plot(windowed_wave, windowed_spectrum, 'r-', linewidth=1, label='Windowed')
    axes[1,1].scatter(windowed_wave, windowed_spectrum)
    axes[1,1].set_xlim(wave_min,wave_max)
    axes[1,1].set_title(f'Windowed Central Spectrum\n(spaxel {center_i}, {center_j})')
    axes[1,1].set_xlabel('Wavelength (Å)')
    axes[1,1].set_ylabel('Flux')
    axes[1,1].grid(True, alpha=0.3)
    
    # 3. Integrated spectra
    orig_total = np.sum(original_data, axis=(0,1))
    windowed_total = np.sum(windowed_data, axis=(0,1))
    
    axes[2,0].plot(orig_wave, orig_total, 'b-', linewidth=1)
    axes[2,0].scatter(orig_wave, orig_total)
    axes[2,0].set_xlim(wave_min,wave_max)
    axes[2,0].set_title('Original Integrated Spectrum')
    axes[2,0].set_xlabel('Wavelength (Å)')
    axes[2,0].set_ylabel('Total Flux')
    axes[2,0].grid(True, alpha=0.3)
    
    # add window regions
    for i, window in enumerate(windows):
        axes[2,0].axvspan(window['actual_r_min'], window['actual_r_max'], 
                         alpha=0.3, color=colors[i])
    
    axes[2,1].plot(windowed_wave, windowed_total, 'r-', linewidth=1)
    axes[2,1].scatter(windowed_wave, windowed_total)
    axes[2,1].set_xlim(wave_min,wave_max)
    axes[2,1].set_title('Windowed Integrated Spectrum')
    axes[2,1].set_xlabel('Wavelength (Å)')
    axes[2,1].set_ylabel('Total Flux')
    axes[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save plot
    plot_file = Path(output_dir) / 'windowed_data_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Comparison plot: {plot_file}")
    
    return fig

def print_summary(metadata, windows, original_data, windowed_data):
    """Print summary of the windowing process."""
    
    print("\n" + "="*60)
    print("WINDOWING SUMMARY")
    print("="*60)
    
    orig_shape = original_data.shape
    wind_shape = windowed_data.shape
    
    print(f"Original data shape: {orig_shape}")
    print(f"Windowed data shape: {wind_shape}")
    print(f"Data reduction: {orig_shape[2]} → {wind_shape[2]} bins ({100*wind_shape[2]/orig_shape[2]:.1f}%)")
    print(f"Memory reduction: ~{orig_shape[2]/wind_shape[2]:.1f}× smaller")
    
    print(f"\nSpatial coverage: unchanged")
    print(f"  {metadata['ni']} × {metadata['nj']} spaxels")
    print(f"  {metadata['x_min']:.1f}″ to {metadata['x_max']:.1f}″ (X)")
    print(f"  {metadata['y_min']:.1f}″ to {metadata['y_max']:.1f}″ (Y)")
    
    print(f"\nWavelength coverage:")
    print(f"  Original: {metadata['r_min']:.1f} - {metadata['r_max']:.1f} Å ({metadata['r_max']-metadata['r_min']:.1f} Å)")
    if windows:
        total_coverage = sum(w['actual_width'] for w in windows)
        print(f"  Windowed: {total_coverage:.1f} Å total ({len(windows)} windows)")
        print(f"  Coverage efficiency: {100*total_coverage/(metadata['r_max']-metadata['r_min']):.1f}%")
    
    print(f"\nWindow details:")
    for i, window in enumerate(windows):
        print(f"  Window {i+1}: {window['name']}")
        print(f"    {window['actual_r_min']:.1f} - {window['actual_r_max']:.1f} Å ({window['n_bins']} bins)")
        print(f"    New position: bins {window['new_start_bin']} - {window['new_end_bin']}")
    
    print("="*60)

def main():
    """Main function."""
    
    # +++++++++++++ Can be Edited +++++++++++++++++

    input_dir = './examples/485885'
    output_dir = './examples/485885-windowed'

    window_size = 5.0   # size (radius; Angstrom) of the window 
                        #(ie. 8 Angstrom = 16 total)

    min_gap = 0.0       # minimum gap before combining windows (Angstrom) 
                        #(ie. min_gap = 5 then windows that are only 5 Ang apart will be combined)

    create_plots = True # set to False to skip plotting
    
    # common emission lines (wavelength in Angstroms)
    emission_lines = {
        'H-beta': 4861.3,
        '[OIII]5007': 5006.8,
        'H-alpha': 6562.81,
        '[NII]6548': 6548.1,
        '[NII]6583': 6583.1,
        '[SII]6717': 6716.4,
        '[SII]6731': 6730.8
    }

    # file paths
    input_dir = Path(input_dir)
    data_file = input_dir / 'data.txt'
    var_file = input_dir / 'var.txt'
    metadata_file = input_dir / 'metadata.txt'

    # +++++++++ End of can be Edited +++++++++++++

    # check input files exist
    for file in [data_file, var_file, metadata_file]:
        if not file.exists():
            print(f"Error: {file} not found")
            sys.exit(1)
    
    print("Blobby3D Multi-Window Data Trimmer")
    print("="*50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Window size: ±{window_size} Å")
    print(f"Minimum gap for combining: {min_gap} Å")
    
    try:
        # load original data
        metadata = load_original_metadata(metadata_file)
        original_data = load_data_cube(data_file, metadata)
        original_var = load_data_cube(var_file, metadata)
        
        # define emission line windows
        print(f"\nDefining wavelength windows (±{window_size} Å around emission lines):")
        windows = define_emission_line_windows(emission_lines, window_size)
        
        # filter windows to those within the data range
        print(f"\nFiltering windows to data coverage:")
        windows = filter_windows_by_coverage(windows, metadata)
        
        if not windows:
            print("No emission lines found within the wavelength range!")
            sys.exit(1)
        
        # combine overlapping windows
        windows = combine_overlapping_windows(windows, min_gap)
        
        # convert to bin indices
        windows = extract_wavelength_indices(windows, metadata)
        
        # extract windowed data
        windowed_data, windowed_var = extract_windowed_data(original_data, original_var, windows)
        
        # create output directory and save data
        output_dir = Path(output_dir)
        write_windowed_data(windowed_data, windowed_var, output_dir)
        write_metadata(metadata, windows, output_dir)
        
        # create comparison plot if requested
        if create_plots:
            print(f"\nCreating comparison plots:")
            create_comparison_plot(original_data, windowed_data, metadata, windows, output_dir)
        
        # print summary
        print_summary(metadata, windows, original_data, windowed_data)
        
        print(f"\nProcessing complete! Output saved to: {output_dir}")
        print(f"Use the new metadata.txt file with your updated Blobby3D code.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()