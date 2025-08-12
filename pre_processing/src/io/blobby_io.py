#!/usr/bin/env python3
"""
Input/output functions for Blobby3D format data with LSF FWHM support.
"""

import numpy as np
from pathlib import Path


def load_blobby_metadata(metadata_path):
    """
    Load Blobby3D metadata from file, supporting both old and new formats.
    
    Old format: Single line with 9 values: Ni Nj Nk x_min x_max y_min y_max wave_min wave_max
    New format: Keyword-value pairs with wave_range specifications (now with lsf_fwhm instead of resolution)
    
    Parameters
    ----------
    metadata_path : str or Path
        Path to metadata.txt file
    
    Returns
    -------
    dict
        Metadata dictionary with spatial and wavelength information
    """
    print(f"Loading Blobby3D metadata from: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        content = f.read().strip()
    
    # detect format by checking if first non-comment line has keywords
    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
    
    if not lines:
        raise ValueError("No data found in metadata file")
    
    first_line = lines[0]
    parts = first_line.split()
    
    # check if it's old format (should be 9 numbers)
    if len(parts) == 9 and all(_is_number(part) for part in parts):
        print("  Detected old format metadata")
        return _load_old_format_metadata(content)
    else:
        print("  Detected new format metadata")
        return _load_new_format_metadata(content)


def _is_number(s):
    """Check if string can be converted to a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _load_old_format_metadata(content):
    """
    Load metadata from old format (single line with 9 values).
    
    Format: Ni Nj Nk x_min x_max y_min y_max wave_min wave_max
    Note: Old format doesn't have LSF FWHM, so it will be None unless provided during conversion
    """
    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
    
    if len(lines) != 1:
        raise ValueError("Old format should have exactly one data line")
    
    parts = lines[0].split()
    if len(parts) != 9:
        raise ValueError(f"Old format requires 9 values, got {len(parts)}")
    
    try:
        ni = int(parts[0])
        nj = int(parts[1])
        nk = int(parts[2])  # total wavelength bins
        x_min = float(parts[3])
        x_max = float(parts[4])
        y_min = float(parts[5])
        y_max = float(parts[6])
        wave_min = float(parts[7])
        wave_max = float(parts[8])
    except ValueError as e:
        raise ValueError(f"Cannot parse old format metadata: {e}")
    
    # create single wavelength range
    wavelength_ranges = [{
        'r_min': wave_min,
        'r_max': wave_max,
        'start_col': 0,
        'end_col': nk - 1,
        'n_bins': nk,
        'lsf_fwhm': None  # no lsf_fwhm in old format
    }]
    
    # create wavelength array (bin centres)
    dr = (wave_max - wave_min) / nk
    wavelengths = np.linspace(wave_min + 0.5*dr, wave_max - 0.5*dr, nk)
    
    metadata = {
        'ni': ni,
        'nj': nj,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'r_min': wave_min,
        'r_max': wave_max,
        'total_bins': nk,
        'wavelength_ranges': wavelength_ranges,
        'wavelengths': wavelengths,
        'format': 'old'
    }
    
    print(f"  Dimensions: {ni} x {nj} x {nk}")
    print(f"  Spatial bounds: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
    print(f"  Wavelength coverage: {wave_min:.2f} - {wave_max:.2f} Å ({nk} bins)")
    print(f"  LSF FWHM: Not available in old format")
    
    return metadata


def _load_new_format_metadata(content):
    """
    Load metadata from new format (keyword-value pairs with wave_range and lsf_fwhm).
    """
    metadata = {}
    wavelength_ranges = []
    
    for line in content.split('\n'):
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
                # check for lsf_fwhm parameter (6th parameter)
                if len(parts) >= 7:
                    wave_range['lsf_fwhm'] = float(parts[6])
                else:
                    wave_range['lsf_fwhm'] = None
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
    metadata['format'] = 'new'
    
    # create combined wavelength array
    wavelengths = []
    for wr in wavelength_ranges:
        dr = (wr['r_max'] - wr['r_min']) / wr['n_bins']
        wave_range = np.linspace(wr['r_min'] + 0.5*dr, wr['r_max'] - 0.5*dr, wr['n_bins'])
        wavelengths.extend(wave_range)
    
    metadata['wavelengths'] = np.array(wavelengths)
    
    print(f"  Dimensions: {metadata['ni']} x {metadata['nj']} x {metadata['total_bins']}")
    print(f"  Wavelength coverage: {metadata['r_min']:.2f} - {metadata['r_max']:.2f} Å")
    print(f"  Number of ranges: {len(wavelength_ranges)}")
    for i, wr in enumerate(wavelength_ranges):
        lsf_str = f" (LSF FWHM={wr['lsf_fwhm']:.3f} Å)" if wr['lsf_fwhm'] is not None else " (No LSF FWHM)"
        print(f"    Range {i+1}: {wr['r_min']:.2f} - {wr['r_max']:.2f} Å ({wr['n_bins']} bins){lsf_str}")
    
    return metadata


def convert_old_to_new_metadata(old_metadata_path, new_metadata_path=None, lsf_fwhm=None, resolution=None):
    """
    Convert old format metadata to new format with LSF FWHM.
    
    Parameters
    ----------
    old_metadata_path : str or Path
        Path to old format metadata.txt file
    new_metadata_path : str or Path, optional
        Path for new format file. If None, adds '_new' suffix to original name.
    lsf_fwhm : float, optional
        LSF FWHM in Angstroms to use for the converted metadata
    resolution : float, optional
        Spectral resolution R = λ/Δλ. If provided and lsf_fwhm is None,
        will calculate LSF FWHM = λ_central / R
    
    Returns
    -------
    Path
        Path to the new metadata file
    """
    old_path = Path(old_metadata_path)
    
    if new_metadata_path is None:
        new_path = old_path.parent / f"{old_path.stem}_new{old_path.suffix}"
    else:
        new_path = Path(new_metadata_path)
    
    print(f"Converting metadata format:")
    print(f"  From: {old_path}")
    print(f"  To: {new_path}")
    
    # load old format
    metadata = load_blobby_metadata(old_metadata_path)
    
    if metadata.get('format') != 'old':
        print("  Warning: Input file is already in new format")
        return old_path
    
    # calculate LSF FWHM if not provided
    if lsf_fwhm is None and resolution is not None:
        # calculate central wavelength of the range
        central_wavelength = (metadata['r_min'] + metadata['r_max']) / 2.0
        lsf_fwhm = central_wavelength / resolution
        print(f"  Calculated LSF FWHM: {lsf_fwhm:.3f} Å (λ_central={central_wavelength:.1f} Å, R={resolution:.0f})")
    elif lsf_fwhm is not None:
        print(f"  Using provided LSF FWHM: {lsf_fwhm:.3f} Å")
    else:
        print("  Warning: No LSF FWHM or resolution provided")
    
    # write in new format
    print(f"  Writing new format metadata...")
    
    with open(new_path, 'w') as f:
        f.write("# Blobby3D Metadata File - New Format with LSF FWHM\n")
        f.write(f"# Converted from old format: {old_path.name}\n")
        if lsf_fwhm is not None:
            f.write(f"# LSF FWHM: {lsf_fwhm:.3f} Å\n")
        if resolution is not None:
            f.write(f"# Original spectral resolution: R = {resolution:.0f}\n")
        f.write("\n")
        
        # spatial dimensions
        f.write(f"ni {metadata['ni']}\n")
        f.write(f"nj {metadata['nj']}\n")
        f.write(f"x_min {metadata['x_min']:.6f}\n")
        f.write(f"x_max {metadata['x_max']:.6f}\n")
        f.write(f"y_min {metadata['y_min']:.6f}\n")
        f.write(f"y_max {metadata['y_max']:.6f}\n")
        
        # wavelength range (single range for old format)
        wr = metadata['wavelength_ranges'][0]
        wave_range_line = f"wave_range {wr['r_min']:.6f} {wr['r_max']:.6f} {wr['start_col']} {wr['end_col']} {wr['n_bins']}"
        
        # add lsf_fwhm if provided
        if lsf_fwhm is not None:
            wave_range_line += f" {lsf_fwhm:.6f}"
        
        wave_range_line += "  # Full wavelength range"
        if lsf_fwhm is not None:
            wave_range_line += f" (LSF FWHM={lsf_fwhm:.3f} Å)"
        
        f.write(wave_range_line + "\n")
        
        f.write(f"\n# Converted from old format with {metadata['total_bins']} wavelength bins\n")
        f.write(f"# Original range: {metadata['r_min']:.2f} - {metadata['r_max']:.2f} Å\n")
        if lsf_fwhm is not None:
            f.write(f"# Format: wave_range r_min r_max start_bin end_bin n_bins lsf_fwhm\n")
        else:
            f.write(f"# Format: wave_range r_min r_max start_bin end_bin n_bins\n")
        f.write(f"# LSF FWHM units: Angstroms\n")
    
    print(f"  Conversion complete!")
    return new_path


def load_blobby_data_cube(data_path, metadata):
    """
    Load and reshape Blobby3D data cube.
    
    Parameters
    ----------
    data_path : str or Path
        Path to data.txt file
    metadata : dict
        Metadata dictionary from load_blobby_metadata
    
    Returns
    -------
    np.ndarray
        Data cube in (ni, nj, nr) format
    """
    print(f"Loading Blobby3D data cube from: {data_path}")
    data_flat = np.loadtxt(data_path)
    
    # reshape from flat array to 3D cube (ni, nj, nr)
    expected_size = metadata['ni'] * metadata['nj'] * metadata['total_bins']
    if len(data_flat.flatten()) != expected_size:
        raise ValueError(f"Data size mismatch: expected {expected_size}, got {len(data_flat.flatten())}")
    
    data_cube = data_flat.reshape(metadata['ni'], metadata['nj'], metadata['total_bins'])
    print(f"  Data cube shape: {data_cube.shape}")
    return data_cube


def write_blobby_data(windowed_data, windowed_weights, windowed_var, output_dir):
    """
    Write windowed data to Blobby3D format text files.
    
    Parameters
    ----------
    windowed_data : np.ndarray
        Data in (ni, nj, nr) format
    windowed_weights : np.ndarray or None
        Weights in same format as data
    windowed_var : np.ndarray or None
        Variance in same format as data
    output_dir : str or Path
        Output directory path
    
    Returns
    -------
    tuple
        (data_file, var_file, weights_file) - Path objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # flatten spatial dimensions
    data_flat = windowed_data.reshape(-1, windowed_data.shape[-1])
    var_flat = windowed_var.reshape(-1, windowed_var.shape[-1]) if windowed_var is not None else None
    weights_flat = windowed_weights.reshape(-1, windowed_weights.shape[-1]) if windowed_weights is not None else None
    
    data_file = output_dir / 'data.txt'
    var_file = output_dir / 'var.txt'
    weights_file = output_dir / 'weights.txt'
    
    print(f"\nWriting Blobby3D data:")
    print(f"  Data: {data_file}")
    
    np.savetxt(data_file, data_flat, fmt='%.8e')
    
    if var_flat is not None:
        print(f"  Variance: {var_file}")
        np.savetxt(var_file, var_flat, fmt='%.8e')
    
    if weights_flat is not None:
        print(f"  Weights: {weights_file}")
        np.savetxt(weights_file, weights_flat, fmt='%.8e')
    
    return data_file, var_file, weights_file


def write_blobby_metadata(coord_info, windows, output_dir, ni, nj):
    """
    Write Blobby3D metadata file with LSF FWHM.
    
    Parameters
    ----------
    coord_info : dict
        Coordinate information dictionary
    windows : list
        List of window dictionaries (should contain 'lsf_fwhm' key)
    output_dir : str or Path
        Output directory path
    ni, nj : int
        Spatial dimensions
    
    Returns
    -------
    Path
        Path to written metadata file
    """
    output_dir = Path(output_dir)
    metadata_file = output_dir / 'metadata.txt'
    
    print(f"  Metadata: {metadata_file}")
    
    with open(metadata_file, 'w') as f:
        f.write("# Blobby3D Metadata File - New Format with LSF FWHM\n")
        f.write("# Generated by IFS Data Processor\n\n")
        
        # spatial dimensions
        f.write(f"ni {ni}\n")
        f.write(f"nj {nj}\n")
        f.write(f"x_min {coord_info['x_min']:.6f}\n")
        f.write(f"x_max {coord_info['x_max']:.6f}\n")
        f.write(f"y_min {coord_info['y_min']:.6f}\n")
        f.write(f"y_max {coord_info['y_max']:.6f}\n")
        
        # wavelength windows with LSF FWHM
        for window in windows:
            lsf_fwhm_str = f" {window['lsf_fwhm']:.6f}" if 'lsf_fwhm' in window and window['lsf_fwhm'] is not None else ""
            f.write(f"wave_range {window['actual_r_min']:.6f} {window['actual_r_max']:.6f} "
                   f"{window['new_start_bin']} {window['new_end_bin']} {window['n_bins']}"
                   f"{lsf_fwhm_str}  # {window['name']}")
            if 'lsf_fwhm' in window and window['lsf_fwhm'] is not None:
                f.write(f" (LSF FWHM={window['lsf_fwhm']:.3f} Å)")
            f.write("\n")
        
        f.write(f"\n# Total wavelength bins: {sum(w['n_bins'] for w in windows)}\n")
        f.write(f"# Format: wave_range r_min r_max start_bin end_bin n_bins [lsf_fwhm] # name\n")
        f.write(f"# LSF FWHM units: Angstroms\n")
        f.write(f"# LSF FWHM = λ_central / R (calculated before redshift correction)\n")
    
    return metadata_file