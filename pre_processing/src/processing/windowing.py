#!/usr/bin/env python3
"""
Updated windowing functions with per-window LSF FWHM calculation.
Each window gets its own LSF FWHM based on its central wavelength and instrument resolution.
"""

import numpy as np
from pathlib import Path


def extract_wavelength_indices_with_per_window_lsf(
    windows, 
    wavelengths, 
    instrument_resolutions,
    instrument_wavelength_ranges,
    redshift=0.0,
    default_resolution=None
):
    """
    Convert wavelength ranges to bin indices and calculate per-window LSF FWHM.
    
    Each window gets its own LSF FWHM calculated as:
    LSF_FWHM = λ_central_observed / R_instrument
    where λ_central_observed = λ_central_rest × (1 + redshift)
    
    Parameters
    ----------
    windows : list
        List of window dictionaries
    wavelengths : np.ndarray
        Array of rest-frame wavelength bin centres (after redshift correction)
    instrument_resolutions : dict
        Dictionary mapping instrument names to their spectral resolutions
        e.g., {'blue': 1800, 'red': 4300}
    instrument_wavelength_ranges : dict
        Dictionary mapping instrument names to their wavelength coverage
        e.g., {'blue': (3700, 5500), 'red': (5500, 7500)}
    redshift : float
        Redshift value used in the correction
    default_resolution : float, optional
        Default resolution to use if instrument cannot be determined
    
    Returns
    -------
    list
        Updated windows with bin indices, actual wavelength ranges, and per-window LSF FWHM
    """
    print(f"\nMapping windows to spectral bins and calculating per-window LSF FWHM:")
    print(f"Redshift: {redshift}")
    print(f"Instrument resolutions: {instrument_resolutions}")
    print(f"Instrument ranges: {instrument_wavelength_ranges}")
    
    for window in windows:
        # find closest indices in the rest-frame wavelength array
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
        
        # calculate central wavelength (rest-frame)
        central_wavelength_rest = (window['actual_r_min'] + window['actual_r_max']) / 2.0
        
        # convert to observed wavelength (before redshift correction)
        central_wavelength_observed = central_wavelength_rest * (1 + redshift)
        
        # determine which instrument covers this wavelength
        covering_instrument = None
        instrument_resolution = None
        
        for instrument, (wave_min, wave_max) in instrument_wavelength_ranges.items():
            if wave_min <= central_wavelength_observed <= wave_max:
                covering_instrument = instrument
                instrument_resolution = instrument_resolutions.get(instrument)
                break
        
        # fallback to overlapping instrument if no exact match
        if covering_instrument is None:
            print(f"    Warning: No exact instrument match for {central_wavelength_observed:.1f} Å")
            # find best overlapping instrument
            best_overlap = 0
            for instrument, (wave_min, wave_max) in instrument_wavelength_ranges.items():
                overlap = max(0, min(wave_max, central_wavelength_observed + 100) - 
                             max(wave_min, central_wavelength_observed - 100))
                if overlap > best_overlap:
                    best_overlap = overlap
                    covering_instrument = instrument
                    instrument_resolution = instrument_resolutions.get(instrument)
        
        # final fallback to default resolution
        if instrument_resolution is None:
            instrument_resolution = default_resolution
            covering_instrument = 'default'
            if default_resolution is None:
                print(f"    Error: No resolution found for {window['name']}")
                window['lsf_fwhm'] = None
                window['instrument'] = 'unknown'
                continue
        
        # calculate LSF FWHM for this specific window
        window['lsf_fwhm'] = central_wavelength_observed / instrument_resolution
        window['instrument'] = covering_instrument
        window['resolution'] = instrument_resolution
        window['central_wavelength_observed'] = central_wavelength_observed
        window['central_wavelength_rest'] = central_wavelength_rest
        
        print(f"  {window['name']}:")
        print(f"    Rest-frame range: {window['actual_r_min']:.1f} - {window['actual_r_max']:.1f} Å")
        print(f"    Central λ (rest): {central_wavelength_rest:.1f} Å")
        print(f"    Central λ (obs): {central_wavelength_observed:.1f} Å")
        print(f"    Instrument: {covering_instrument} (R = {instrument_resolution})")
        print(f"    LSF FWHM: {window['lsf_fwhm']:.3f} Å")
        print(f"    Bins: {start_idx} - {end_idx} ({window['n_bins']} bins)")
    
    return windows


def extract_wavelength_indices_single_instrument(
    windows, 
    wavelengths, 
    resolution,
    redshift=0.0,
    lsf_fwhm=None
):
    """
    Simplified version for single instrument data.
    
    Parameters
    ----------
    windows : list
        List of window dictionaries
    wavelengths : np.ndarray
        Array of rest-frame wavelength bin centres
    resolution : float
        Spectral resolution R = λ/Δλ
    redshift : float
        Redshift value
    lsf_fwhm : float, optional
        If provided, use this fixed LSF FWHM for all windows instead of calculating
    
    Returns
    -------
    list
        Updated windows with per-window LSF FWHM
    """
    print(f"\nMapping windows for single instrument (R = {resolution}):")
    
    for window in windows:
        # standard index calculation
        start_idx = np.argmin(np.abs(wavelengths - window['r_min']))
        end_idx = np.argmin(np.abs(wavelengths - window['r_max']))
        
        if end_idx < start_idx:
            end_idx = start_idx
        
        window['start_idx'] = start_idx
        window['end_idx'] = end_idx
        window['n_bins'] = end_idx - start_idx + 1
        
        # calculate actual range
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
        
        # calculate central wavelengths
        central_wavelength_rest = (window['actual_r_min'] + window['actual_r_max']) / 2.0
        central_wavelength_observed = central_wavelength_rest * (1 + redshift)
        
        # calculate per-window LSF FWHM
        if lsf_fwhm is not None:
            # use provided fixed LSF FWHM
            window['lsf_fwhm'] = lsf_fwhm
        else:
            # calculate LSF FWHM for this specific window
            window['lsf_fwhm'] = central_wavelength_observed / resolution
        
        window['instrument'] = 'single'
        window['resolution'] = resolution
        window['central_wavelength_observed'] = central_wavelength_observed
        window['central_wavelength_rest'] = central_wavelength_rest
        
        print(f"  {window['name']}:")
        print(f"    Central λ (obs): {central_wavelength_observed:.1f} Å")
        print(f"    LSF FWHM: {window['lsf_fwhm']:.3f} Å")
    
    return windows


def propagate_per_window_lsf_from_existing_metadata(windows, metadata):
    """
    Propagate per-window LSF FWHM from existing metadata when re-windowing.
    
    For existing data, we don't know the original redshift and instrument setup,
    so we need to handle LSF FWHM differently:
    - If existing metadata has per-window LSF FWHM, use those values as reference
    - If not available, use a default approach
    
    Parameters
    ----------
    windows : list
        List of new window dictionaries
    metadata : dict
        Existing metadata containing wavelength ranges
    
    Returns
    -------
    list
        Updated windows with LSF FWHM values
    """
    print(f"\nPropagating per-window LSF FWHM from existing metadata:")
    
    # collect existing LSF FWHM values and their wavelength ranges
    existing_lsf_data = []
    for wr in metadata['wavelength_ranges']:
        if 'lsf_fwhm' in wr and wr['lsf_fwhm'] is not None:
            central_wave = (wr['r_min'] + wr['r_max']) / 2.0
            existing_lsf_data.append({
                'wavelength': central_wave,
                'lsf_fwhm': wr['lsf_fwhm'],
                'range': (wr['r_min'], wr['r_max'])
            })
    
    if not existing_lsf_data:
        print("  No LSF FWHM values found in existing metadata")
        for window in windows:
            window['lsf_fwhm'] = None
        return windows
    
    print(f"  Found {len(existing_lsf_data)} existing LSF FWHM values:")
    for data in existing_lsf_data:
        print(f"    λ={data['wavelength']:.1f} Å: LSF FWHM={data['lsf_fwhm']:.3f} Å")
    
    # assign LSF FWHM to new windows
    for window in windows:
        central_wave = (window['actual_r_min'] + window['actual_r_max']) / 2.0
        
        # find the closest existing LSF FWHM value by wavelength
        distances = [abs(central_wave - data['wavelength']) for data in existing_lsf_data]
        closest_idx = np.argmin(distances)
        closest_data = existing_lsf_data[closest_idx]
        
        # use the closest LSF FWHM value
        window['lsf_fwhm'] = closest_data['lsf_fwhm']
        
        print(f"  {window['name']} (λ={central_wave:.1f} Å):")
        print(f"    Using LSF FWHM from λ={closest_data['wavelength']:.1f} Å: {window['lsf_fwhm']:.3f} Å")
    
    return windows


# wrapper function to choose the right approach
def extract_wavelength_indices(
    windows, 
    wavelengths, 
    multi_instrument_setup=None,
    resolution=None,
    lsf_fwhm=None,
    redshift=0.0
):
    """
    Wrapper function that chooses the appropriate LSF FWHM calculation method.
    
    Parameters
    ----------
    windows : list
        List of window dictionaries
    wavelengths : np.ndarray
        Array of wavelength bin centres
    multi_instrument_setup : dict, optional
        Dictionary with 'resolutions' and 'wavelength_ranges' for multi-instrument data
    resolution : float, optional
        Single resolution value for single-instrument data
    lsf_fwhm : float, optional
        Fixed LSF FWHM value (overrides resolution calculation)
    redshift : float
        Redshift value
    
    Returns
    -------
    list
        Updated windows with LSF FWHM
    """
    if multi_instrument_setup is not None:
        # multi-instrument case
        return extract_wavelength_indices_with_per_window_lsf(
            windows,
            wavelengths,
            multi_instrument_setup['resolutions'],
            multi_instrument_setup['wavelength_ranges'],
            redshift=redshift
        )
    else:
        # single instrument case
        if resolution is None and lsf_fwhm is None:
            raise ValueError("Must provide either resolution or lsf_fwhm for single instrument")
        
        return extract_wavelength_indices_single_instrument(
            windows,
            wavelengths, 
            resolution or 3000,  # default resolution if calculating from lsf_fwhm
            redshift=redshift,
            lsf_fwhm=lsf_fwhm
        )


# keep the other functions unchanged
def define_emission_line_windows(emission_lines, window_size=35):
    """Define wavelength windows around emission lines."""
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
    """Filter windows to only include those within the wavelength coverage."""
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
        if (window['r_max'] > r_min_total and window['r_min'] < r_max_total):
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
    """Combine overlapping or closely spaced windows."""
    if not windows:
        return []
    
    sorted_windows = sorted(windows, key=lambda x: x['r_min'])
    
    combined = []
    current = sorted_windows[0].copy()
    current['lines'] = [current['name']]
    
    for window in sorted_windows[1:]:
        if window['r_min'] <= current['r_max'] + min_gap:
            current['r_max'] = max(current['r_max'], window['r_max'])
            current['lines'].append(window['name'])
            current['name'] = ' + '.join(current['lines'])
            current['width'] = current['r_max'] - current['r_min']
        else:
            combined.append(current)
            current = window.copy()
            current['lines'] = [current['name']]
    
    combined.append(current)
    
    print(f"\nCombined {len(windows)} windows into {len(combined)} non-overlapping windows:")
    for i, window in enumerate(combined):
        print(f"  Window {i+1}: {window['name']}")
        print(f"    Range: {window['r_min']:.1f} - {window['r_max']:.1f} Å ({window['width']:.1f} Å wide)")
        if 'lines' in window:
            print(f"    Lines: {', '.join(window['lines'])}")
    
    return combined


def extract_windowed_data(data_cube, windows, var_cube=None):
    """Extract data for the specified wavelength windows."""
    nr_orig, ni, nj = data_cube.shape
    
    total_bins = sum(window['n_bins'] for window in windows)
    
    windowed_data = np.zeros((ni, nj, total_bins))
    windowed_var = np.zeros((ni, nj, total_bins)) if var_cube is not None else None
    
    current_bin = 0
    for window in windows:
        start_idx = window['start_idx']
        end_idx = window['end_idx']
        n_bins = window['n_bins']
        
        data_slice = data_cube[start_idx:end_idx+1, :, :].transpose(1, 2, 0)
        windowed_data[:, :, current_bin:current_bin+n_bins] = data_slice
        
        if var_cube is not None:
            var_slice = var_cube[start_idx:end_idx+1, :, :].transpose(1, 2, 0)
            windowed_var[:, :, current_bin:current_bin+n_bins] = var_slice
        
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