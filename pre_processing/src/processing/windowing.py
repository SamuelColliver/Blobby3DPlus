#!/usr/bin/env python3
"""
Wavelength windowing functions for emission line analysis.
"""

import numpy as np
from pathlib import Path


def define_emission_line_windows(emission_lines, window_size=35):
    """
    Define wavelength windows around emission lines.
    
    Parameters
    ----------
    emission_lines : dict
        Dictionary mapping line names to central wavelengths in Angstroms
    window_size : float
        Half-width of window in Angstroms (total width = 2 * window_size)
    
    Returns
    -------
    list
        List of window dictionaries with keys: name, center, r_min, r_max, width
    """
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
    
    print(f"Defined {len(windows)} emission line windows (±{window_size} Å):")
    for window in windows:
        print(f"  {window['name']}: {window['r_min']:.1f} - {window['r_max']:.1f} Å")
    
    return windows


def filter_windows_by_coverage(windows, wavelengths):
    """
    Filter windows to only include those within the wavelength coverage.
    
    Parameters
    ----------
    windows : list
        List of window dictionaries
    wavelengths : np.ndarray or dict
        Wavelength array or dict with r_min/r_max keys
    
    Returns
    -------
    list
        Filtered list of windows
    """
    # handle different input types for wavelength coverage
    if isinstance(wavelengths, dict):
        r_min_total = wavelengths['r_min']
        r_max_total = wavelengths['r_max']
    else:
        r_min_total = np.min(wavelengths)
        r_max_total = np.max(wavelengths)
    
    print(f"\nFiltering windows by wavelength coverage:")
    print(f"Available range: {r_min_total:.1f} - {r_max_total:.1f} Å")
    
    filtered_windows = []
    for window in windows:
        # check if window overlaps with available range
        if (window['r_max'] > r_min_total and window['r_min'] < r_max_total):
            # clip window to available range
            clipped_window = window.copy()
            clipped_window['r_min'] = max(window['r_min'], r_min_total)
            clipped_window['r_max'] = min(window['r_max'], r_max_total)
            clipped_window['width'] = clipped_window['r_max'] - clipped_window['r_min']
            
            if clipped_window['width'] > 0:
                filtered_windows.append(clipped_window)
                print(f"  Including {window['name']}: {clipped_window['r_min']:.1f} - {clipped_window['r_max']:.1f} Å")
            else:
                print(f"  Excluding {window['name']}: zero width after clipping")
        else:
            print(f"  Excluding {window['name']}: outside wavelength range")
    
    return filtered_windows


def combine_overlapping_windows(windows, min_gap=5.0):
    """
    Combine overlapping or closely spaced windows.
    
    Parameters
    ----------
    windows : list
        List of window dictionaries
    min_gap : float
        Minimum gap in Angstroms before combining windows
    
    Returns
    -------
    list
        Combined list of windows
    """
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


def extract_wavelength_indices(windows, wavelengths, resolution=None):
    """
    Convert wavelength ranges to bin indices and update window properties.
    
    Parameters
    ----------
    windows : list
        List of window dictionaries
    wavelengths : np.ndarray
        Array of wavelength bin centres
    resolution : float or dict, optional
        Spectral resolution (R = λ/Δλ). Can be single value or dict with 
        instrument-specific values for multi-instrument data.
    
    Returns
    -------
    list
        Updated windows with bin indices and actual wavelength ranges
    """
    print(f"\nMapping windows to spectral bins:")
    
    for window in windows:
        # find closest indices in the wavelength array
        start_idx = np.argmin(np.abs(wavelengths - window['r_min']))
        end_idx = np.argmin(np.abs(wavelengths - window['r_max']))
        
        # ensure end_idx >= start_idx
        if end_idx < start_idx:
            end_idx = start_idx
        
        window['start_idx'] = start_idx
        window['end_idx'] = end_idx
        window['n_bins'] = end_idx - start_idx + 1
        
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
        window['actual_width'] = window['actual_r_max'] - window['actual_r_min']
        
        # assign resolution if provided
        if resolution is not None:
            if isinstance(resolution, dict):
                # for multi-instrument data, this would need more logic
                # for now, just use a default value
                window['resolution'] = list(resolution.values())[0]
            else:
                window['resolution'] = resolution
        
        print(f"  {window['name']}:")
        print(f"    Requested: {window['r_min']:.1f} - {window['r_max']:.1f} Å")
        print(f"    Actual: {window['actual_r_min']:.1f} - {window['actual_r_max']:.1f} Å")
        print(f"    Bins: {start_idx} - {end_idx} ({window['n_bins']} bins)")
        if resolution is not None:
            print(f"    Resolution: {window.get('resolution', 'N/A')}")
    
    return windows


def extract_windowed_data(data_cube, windows, var_cube=None):
    """
    Extract data for the specified wavelength windows.
    
    Parameters
    ----------
    data_cube : np.ndarray
        Data cube in (wavelength, y, x) format
    windows : list
        List of window dictionaries with bin indices
    var_cube : np.ndarray, optional
        Variance cube in same format as data_cube
    
    Returns
    -------
    tuple
        (windowed_data, windowed_var) in (y, x, wavelength) format for Blobby3D
    """
    nr_orig, ni, nj = data_cube.shape
    
    # calculate total number of bins in windowed data
    total_bins = sum(window['n_bins'] for window in windows)
    
    # create new data cubes in (y, x, wavelength) format for Blobby3D
    windowed_data = np.zeros((ni, nj, total_bins))
    windowed_var = np.zeros((ni, nj, total_bins)) if var_cube is not None else None
    
    # extract data for each window
    current_bin = 0
    for window in windows:
        start_idx = window['start_idx']
        end_idx = window['end_idx']
        n_bins = window['n_bins']
        
        # transpose from (wavelength, y, x) to (y, x, wavelength)
        data_slice = data_cube[start_idx:end_idx+1, :, :].transpose(1, 2, 0)
        windowed_data[:, :, current_bin:current_bin+n_bins] = data_slice
        
        if var_cube is not None:
            var_slice = var_cube[start_idx:end_idx+1, :, :].transpose(1, 2, 0)
            windowed_var[:, :, current_bin:current_bin+n_bins] = var_slice
        
        # update window with position in new array
        window['new_start_bin'] = current_bin
        window['new_end_bin'] = current_bin + n_bins - 1
        
        current_bin += n_bins
    
    print(f"\nExtracted windowed data:")
    print(f"  Original shape: {data_cube.shape} (wavelength, y, x)")
    print(f"  Windowed shape: {windowed_data.shape} (y, x, wavelength)")
    print(f"  Data reduction: {nr_orig} → {total_bins} bins ({100*total_bins/nr_orig:.1f}%)")
    
    return windowed_data, windowed_var


# common emission line definitions
COMMON_EMISSION_LINES = {
    # optical lines
    '[OII]3726': 3726.03,
    '[OII]3729': 3728.82,
    'H-beta': 4861.3,
    '[OIII]4959': 4958.9,
    '[OIII]5007': 5006.8,
    'H-alpha': 6562.81,
    '[NII]6548': 6548.1,
    '[NII]6583': 6583.1,
    '[SII]6717': 6716.4,
    '[SII]6731': 6730.8,
    
    # near-infrared lines
    '[SIII]9069': 9068.6,
    '[SIII]9532': 9531.1,
    'Pa-gamma': 10938.1,
    'Pa-beta': 12818.1,
    'Pa-alpha': 18751.0,
    'Br-gamma': 21655.3,
}