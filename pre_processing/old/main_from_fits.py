#!/usr/bin/env pithon3
"""
Combined SAMI IFS FITS Data Processor for Blobby3D
Processes both blue and red FITS files containing SAMI IFS data,
applies redshift correction, continuum subtraction, data cleaning,
and combines them into windowed emission line data.
"""

import numpy as np
from astropy.io import fits
from astropy import constants as const
import matplotlib.pyplot as plt
import sys
import warnings
from pathlib import Path

# suppress FITS warnings for cleaner output
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

def read_sami_fits(filename):
    """
    Read IFS data from FITS file including data, weights, and variance.
    Reads dimensions from header and handles the transpose correctly.
    """
    print(f"Reading FITS file: {filename}")
    
    with fits.open(filename) as hdul:
        print(f"Found {len(hdul)} extensions in FITS file")
        
        # print extension info for debugging
        for i, hdu in enumerate(hdul):
            ext_name = hdu.header.get('EXTNAME', f'Extension {i}')
            if hdu.data is not None:
                print(f"  Extension {i} ({ext_name}): {hdu.data.shape}")
            else:
                print(f"  Extension {i} ({ext_name}): No data")
        
        # find extensions with 3D data
        data_ext = None
        weights_ext = None
        var_ext = None
        
        for i, hdu in enumerate(hdul):
            if hdu.data is not None and len(hdu.data.shape) == 3:
                ext_name = hdu.header.get('EXTNAME', '').upper()
                
                # look for data extension (usually first 3D extension or named DATA)
                if data_ext is None and ('PRIMARY' in ext_name):
                    data_ext = i
                    print(f"  -> Found data in extension {i}")
                
                # look for weights extension
                elif 'WEIGHT' in ext_name:
                    weights_ext = i
                    print(f"  -> Found weights in extension {i}")
                
                # look for variance extension
                elif 'VARIANCE' in ext_name:
                    var_ext = i
                    print(f"  -> Found variance in extension {i}")
        
        # if data_ext is still None, use first 3D extension
        if data_ext is None:
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) == 3:
                    data_ext = i
                    print(f"  -> Using extension {i} as data (first 3D extension)")
                    break
        
        if data_ext is None:
            raise ValueError("No 3D data cube found in FITS file")
        
        # read data
        data = hdul[data_ext].data
        header = hdul[data_ext].header
        
        # read weights if available
        weights = None
        if weights_ext is not None:
            weights = hdul[weights_ext].data
            print(f"  -> Read weights from extension {weights_ext}")
        
        # read variance if available
        variance = None
        if var_ext is not None:
            variance = hdul[var_ext].data
            print(f"  -> Read variance from extension {var_ext}")
        
        # if weights/variance weren't found by name, try to find them by position
        if weights is None or variance is None:
            print("  -> Searching for weights/variance by position...")
            extensions_3d = []
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) == 3:
                    extensions_3d.append(i)
            
            # common patterns: data, weights, variance OR data, variance, weights
            if len(extensions_3d) >= 3:
                if weights is None and len(extensions_3d) > 1:
                    weights_ext = extensions_3d[1]  # second 3D extension
                    weights = hdul[weights_ext].data
                    print(f"  -> Assuming weights in extension {weights_ext}")
                
                if variance is None and len(extensions_3d) > 2:
                    var_ext = extensions_3d[2]  # third 3D extension
                    variance = hdul[var_ext].data
                    print(f"  -> Assuming variance in extension {var_ext}")
        
        # read dimensions from header
        naxis1 = header['NAXIS1']  # x dimension (expected)
        naxis2 = header['NAXIS2']  # y dimension (expected)
        naxis3 = header['NAXIS3']  # wavelength dimension (expected)
        
        print(f"Header dimensions: NAXIS1={naxis1}, NAXIS2={naxis2}, NAXIS3={naxis3}")
        print(f"Original FITS data shape: {data.shape}")
        
        # function to transpose data to correct orientation
        def transpose_data(data_array, label="data"):
            """Transpose data array to (wavelength, y, x) format"""
            if data_array is None:
                return None
                
            # FITS standard is (NAXIS3, NAXIS2, NAXIS1) = (wavelength, y, x)
            expected_shape = (naxis3, naxis2, naxis1)
            
            if not data_array.shape == expected_shape:
                if data_array.shape == (naxis1, naxis2, naxis3):  # (x, y, wavelength)
                    data_array = np.transpose(data_array, (2, 1, 0))  # -> (wavelength, y, x)
                elif data_array.shape == (naxis2, naxis1, naxis3):  # (y, x, wavelength)
                    data_array = np.transpose(data_array, (2, 0, 1))  # -> (wavelength, y, x)
                elif data_array.shape == (naxis3, naxis1, naxis2):  # (wavelength, x, y)
                    data_array = np.transpose(data_array, (0, 2, 1))  # -> (wavelength, y, x)
                else:
                    # try to auto-detect by finding the wavelength axis (largest dimension)
                    wavelength_axis = np.argmax(data_array.shape)
                    if wavelength_axis == 0:
                        pass  # already correct
                    elif wavelength_axis == 1:
                        data_array = np.transpose(data_array, (1, 0, 2))
                    elif wavelength_axis == 2:
                        data_array = np.transpose(data_array, (2, 0, 1))
                
                print(f"  -> Transposed {label} to shape: {data_array.shape}")
            
            return data_array
        
        # transpose all arrays to correct orientation
        data = transpose_data(data, "data")
        weights = transpose_data(weights, "weights")
        variance = transpose_data(variance, "variance")
        
        # create wavelength array from header WCS
        # use CRVAL3, CDELT3, CRPIX3 for wavelength axis
        if 'CRVAL3' in header and 'CDELT3' in header and 'CRPIX3' in header:
            crval3 = header['CRVAL3']  # reference wavelength
            cdelt3 = header['CDELT3']  # wavelength increment  
            crpix3 = header['CRPIX3']  # reference pixel
            
            # create wavelength array
            pixels = np.arange(1, naxis3 + 1)  # FITS pixels start at 1
            wavelengths = crval3 + cdelt3 * (pixels - crpix3)
            
            print(f"Using CRVAL3/CDELT3/CRPIX3 wavelength solution:")
            print(f"  CRVAL3={crval3:.2f} Å, CDELT3={cdelt3:.4f} Å/pixel, CRPIX3={crpix3}")
            
        else:
            print("Warning: No standard wavelength WCS found in header")
            wavelengths = np.arange(naxis3, dtype=float)
    
    print(f"Final data shape: {data.shape}")
    if weights is not None:
        print(f"Final weights shape: {weights.shape}")
    if variance is not None:
        print(f"Final variance shape: {variance.shape}")
    print(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} Å")
    print(f"Number of wavelength channels: {len(wavelengths)}")
    
    return data, weights, variance, header, wavelengths

def redshift_correct_wavelengths(wavelengths, redshift):
    """
    Apply redshift correction to wavelengths.
    """
    if redshift == 0:
        return wavelengths
    
    rest_wavelengths = wavelengths / (1 + redshift)
    print(f"Rest-frame wavelength range: {rest_wavelengths[0]:.2f} - {rest_wavelengths[-1]:.2f} Å")
    
    return rest_wavelengths

def subtract_continuum_nanmedian(data, weights=None):
    """
    Subtract continuum using nanmedian across wavelength axis.
    For weights, continuum subtraction doesn't apply - weights remain unchanged.
    """
    print("Subtracting continuum using nanmedian...")
    
    # calculate continuum (median along wavelength axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        continuum = np.nanmedian(data, axis=0, keepdims=True)
    
    # replace NaN continuum values with 0 to avoid issues
    continuum = np.nan_to_num(continuum, nan=0.0)
    
    # subtract continuum from data
    continuum_subtracted = data - continuum
    
    return continuum_subtracted

def subtract_continuum(data, continuum, weights):
    """
    Subtract continuum.
    """
    print("Subtracting continuum...")
    
    if np.shape(data) == np.shape(continuum):
        return data - continuum
    else:
        print(f'ERROR: data shape {np.shape(data)} does not match contiuum {np.shape(continuum)}. Returning median subtracted.')
        return subtract_continuum_nanmedian(data, weights)
    
def read_continuum(filename, extension_name):
    """
    Read IFS data from FITS file for continuum.
    """
    print(f"Reading FITS file: {filename}")
    
    with fits.open(filename) as hdul:
        # read data
        conti = hdul[extension_name].data
        header = hdul[extension_name].header

        # read dimensions from header
        naxis1 = header['NAXIS1']  # x dimension (expected)
        naxis2 = header['NAXIS2']  # y dimension (expected)
        naxis3 = header['NAXIS3']  # wavelength dimension (expected)
        
        print(f"Header dimensions: NAXIS1={naxis1}, NAXIS2={naxis2}, NAXIS3={naxis3}")
        print(f"Original FITS continuum shape: {conti.shape}")
        
        # function to transpose data to correct orientation
        def transpose_data(data_array, label="data"):
            """Transpose data array to (wavelength, y, x) format"""
            if data_array is None:
                return None
                
            # FITS standard is (NAXIS3, NAXIS2, NAXIS1) = (wavelength, y, x)
            expected_shape = (naxis3, naxis2, naxis1)
            
            if not data_array.shape == expected_shape:
                if data_array.shape == (naxis1, naxis2, naxis3):  # (x, y, wavelength)
                    data_array = np.transpose(data_array, (2, 1, 0))  # -> (wavelength, y, x)
                elif data_array.shape == (naxis2, naxis1, naxis3):  # (y, x, wavelength)
                    data_array = np.transpose(data_array, (2, 0, 1))  # -> (wavelength, y, x)
                elif data_array.shape == (naxis3, naxis1, naxis2):  # (wavelength, x, y)
                    data_array = np.transpose(data_array, (0, 2, 1))  # -> (wavelength, y, x)
                else:
                    # try to auto-detect by finding the wavelength axis (largest dimension)
                    wavelength_axis = np.argmax(data_array.shape)
                    if wavelength_axis == 0:
                        pass  # already correct
                    elif wavelength_axis == 1:
                        data_array = np.transpose(data_array, (1, 0, 2))
                    elif wavelength_axis == 2:
                        data_array = np.transpose(data_array, (2, 0, 1))
                
                print(f"  -> Transposed {label} to shape: {data_array.shape}")
            
            return data_array
        
        # transpose all arrays to correct orientation
        conti = transpose_data(conti, "continuum")
    
    print(f"Final continuum shape: {conti.shape}")
    
    return conti

def clean_data(data, pixelscale_arcsec, weights=None, variance=None):
    """
    Remove rows and columns from edges where all wavelengths are NaN,
    effectively cropping the data to its valid bounds.
    Apply the same cropping to weights and variance.
    Calculate coordinate ranges based on pixel scale with centre at 0.0.
    """
    print("Cleaning data - cropping NaN edges...")
    
    # store original dimensions for coordinate calculation
    original_height, original_width = data.shape[-2], data.shape[-1]
    
    # for 3D data (wavelength, y, x), check if all wavelengths are NaN for each spatial position
    if data.ndim == 3:
        # check for positions where all wavelengths are NaN
        all_nan_spatial = np.all(np.isnan(data), axis=0)  # shape: (y, x)
    else:
        # for 2D data, directly check for NaN
        all_nan_spatial = np.isnan(data)
    
    # find valid bounds
    # get indices where we have valid (non-NaN) data
    valid_rows = ~np.all(all_nan_spatial, axis=1)  # rows that have at least one valid value
    valid_cols = ~np.all(all_nan_spatial, axis=0)  # columns that have at least one valid value
    
    # find the bounding box of valid data
    valid_row_indices = np.where(valid_rows)[0]
    valid_col_indices = np.where(valid_cols)[0]
    
    if len(valid_row_indices) == 0 or len(valid_col_indices) == 0:
        print("Warning: No valid data found!")
        return data, weights, variance, np.zeros_like(all_nan_spatial, dtype=bool), {
            'x_min': 0.0,
            'x_max': 0.0,
            'y_min': 0.0,
            'y_max': 0.0,
            'spatial_sampling': pixelscale_arcsec,
        }
    
    # get the bounds
    row_start, row_end = valid_row_indices[0], valid_row_indices[-1] + 1
    col_start, col_end = valid_col_indices[0], valid_col_indices[-1] + 1
    
    # calculate coordinate ranges
    # original coordinate ranges (centre pixel at 0.0)
    original_x_half_range = (original_width * pixelscale_arcsec) / 2.0
    original_y_half_range = (original_height * pixelscale_arcsec) / 2.0
    original_x_range = (-original_x_half_range, original_x_half_range)
    original_y_range = (-original_y_half_range, original_y_half_range)
    
    # calculate how many pixels were removed from each edge
    rows_removed_top = row_start
    rows_removed_bottom = original_height - row_end
    cols_removed_left = col_start
    cols_removed_right = original_width - col_end
    
    # calculate new coordinate ranges after cropping
    # x-coordinates: left edge moves right by removed pixels * scale
    new_x_left = original_x_range[0] + (cols_removed_left * pixelscale_arcsec)
    new_x_right = original_x_range[1] - (cols_removed_right * pixelscale_arcsec)
    
    # y-coordinates: bottom edge moves up by removed pixels * scale  
    # (assuming y increases upward, row 0 is bottom)
    new_y_bottom = original_y_range[0] + (rows_removed_bottom * pixelscale_arcsec)
    new_y_top = original_y_range[1] - (rows_removed_top * pixelscale_arcsec)
    
    cropped_x_range = (new_x_left, new_x_right)
    cropped_y_range = (new_y_bottom, new_y_top)
    
    # crop the data
    if data.ndim == 3:
        cropped_data = data[:, row_start:row_end, col_start:col_end]
        # apply same cropping to weights and variance
        cropped_weights = weights[:, row_start:row_end, col_start:col_end] if weights is not None else None
        cropped_variance = variance[:, row_start:row_end, col_start:col_end] if variance is not None else None
        
        # update the valid mask to match the cropped dimensions
        valid_mask = ~np.all(np.isnan(cropped_data), axis=0)
    else:
        cropped_data = data[row_start:row_end, col_start:col_end]
        cropped_weights = weights[row_start:row_end, col_start:col_end] if weights is not None else None
        cropped_variance = variance[row_start:row_end, col_start:col_end] if variance is not None else None
        valid_mask = ~np.isnan(cropped_data)
    
    # calculate statistics
    original_size = original_height * original_width
    cropped_size = cropped_data.shape[-2] * cropped_data.shape[-1]
    removed_count = original_size - cropped_size
    
    # create coordinate info dictionary
    coord_info = {
        'x_min': new_x_left,
        'x_max': new_x_right,
        'y_min': new_y_bottom,
        'y_max': new_y_top,
        'spatial_sampling': pixelscale_arcsec
    }
    
    print(f"Original dimensions: {data.shape}")
    print(f"Cropped dimensions: {cropped_data.shape}")
    if cropped_weights is not None:
        print(f"Cropped weights dimensions: {cropped_weights.shape}")
    if cropped_variance is not None:
        print(f"Cropped variance dimensions: {cropped_variance.shape}")
    print(f"Cropped from (rows {row_start}:{row_end}, cols {col_start}:{col_end})")
    print(f"Removed {removed_count} edge spaxels out of {original_size} total")
    print(f"Pixel scale: {pixelscale_arcsec} arcsec/pixel")
    print(f"Original coordinate ranges:")
    print(f"  X: {original_x_range[0]:.2f} to {original_x_range[1]:.2f} arcsec")
    print(f"  Y: {original_y_range[0]:.2f} to {original_y_range[1]:.2f} arcsec")
    print(f"Cropped coordinate ranges:")
    print(f"  X: {cropped_x_range[0]:.2f} to {cropped_x_range[1]:.2f} arcsec")
    print(f"  Y: {cropped_y_range[0]:.2f} to {cropped_y_range[1]:.2f} arcsec")
    print(f"Pixels removed: {rows_removed_top} top, {rows_removed_bottom} bottom, {cols_removed_left} left, {cols_removed_right} right")
    
    return cropped_data, cropped_weights, cropped_variance, valid_mask, coord_info

def process_single_fits(input_file, pix_size_arcsec, redshift, remove_invalid_spaxels=True, continuum_file = None, continuum_extension_name = None):
    """Process a single FITS file and return cleaned data."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"{'='*60}")
    
    # read SAMI FITS data including weights and variance
    print("Step 1: Reading SAMI FITS file...")
    data, weights, variance, header, wavelengths = read_sami_fits(input_file)
    
    print(f"  - Data cube shape: {data.shape}")
    print(f"  - Weights available: {'Yes' if weights is not None else 'No'}")
    print(f"  - Variance available: {'Yes' if variance is not None else 'No'}")
    print(f"  - Instrument: {header.get('INSTRUME', 'Unknown')}")
    print(f"  - Plate ID: {header.get('PLATEID', 'Unknown')}")
    print(f"  - Grating: {header.get('GRATID', 'Unknown')}")
    print(f"  - Data units: {header.get('BUNIT', 'Unknown')}")
    print(f"  - Original wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} Å")
    
    # apply redshift correction
    if redshift != 0.0:
        print("\nStep 2: Applying redshift correction...")
        rest_wavelengths = redshift_correct_wavelengths(wavelengths, redshift)
        print(f"  - Rest-frame wavelength range: {rest_wavelengths[0]:.2f} - {rest_wavelengths[-1]:.2f} Å")
    else:
        print("\nStep 2: No redshift correction applied (z=0)")
        rest_wavelengths = wavelengths
    
    # subtract continuum (only from data, not weights/variance)
    print("\nStep 3: Subtracting continuum...")
    if continuum_file and continuum_extension_name:
        continuum = read_continuum(continuum_file, continuum_extension_name)
        continuum_subtracted = subtract_continuum(data, continuum, weights)
        print(f"  - Using continuum from file")
    else:
        continuum_subtracted = subtract_continuum_nanmedian(data, weights)
        print(f"  - Using nanmedian across wavelength axis")
    print(f"  - Data range after continuum subtraction: {np.nanmin(continuum_subtracted):.2e} to {np.nanmax(continuum_subtracted):.2e}")
    
    # clean data (crop all arrays consistently)
    if remove_invalid_spaxels:
        print("\nStep 4: Cleaning data...")
        cleaned_data, cleaned_weights, cleaned_variance, valid_mask, coord_info = clean_data(
            continuum_subtracted, pix_size_arcsec, weights, variance)
        total_spaxels = data.shape[1] * data.shape[2]
        valid_spaxels = np.sum(valid_mask)
        print(f"  - Valid spaxels: {valid_spaxels}/{total_spaxels} ({100*valid_spaxels/total_spaxels:.1f}%)")
    else:
        print("\nStep 4: Skipping data cleaning (keeping all spaxels)")
        cleaned_data = continuum_subtracted
        cleaned_weights = weights
        cleaned_variance = variance
        height, width = data.shape[-2], data.shape[-1]
        coord_info = {
            'x_min': -(width * pix_size_arcsec) / 2.0,
            'x_max': (width * pix_size_arcsec) / 2.0,
            'y_min': -(height * pix_size_arcsec) / 2.0,
            'y_max': (height * pix_size_arcsec) / 2.0,
            'spatial_sampling': pix_size_arcsec
        }
        valid_mask = np.ones((data.shape[1], data.shape[2]), dtype=bool)
    
    return {
        'data': cleaned_data,
        'weights': cleaned_weights,
        'variance': cleaned_variance,
        'wavelengths': rest_wavelengths,
        'coord_info': coord_info,
        'header': header
    }

def define_emission_line_windows(emission_lines, window_size=35):
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

def filter_windows_by_coverage(windows, combined_wavelengths):
    """Filter windows to only include those within the combined wavelength range."""
    
    r_min_combined = np.min(combined_wavelengths)
    r_max_combined = np.max(combined_wavelengths)
    
    print(f"\nFiltering windows by combined wavelength coverage:")
    print(f"Combined range: {r_min_combined:.1f} - {r_max_combined:.1f} Å")
    
    filtered_windows = []
    for window in windows:
        # check if window overlaps with combined range
        if (window['r_max'] > r_min_combined and window['r_min'] < r_max_combined):
            # clip window to combined range
            clipped_window = window.copy()
            clipped_window['r_min'] = max(window['r_min'], r_min_combined)
            clipped_window['r_max'] = min(window['r_max'], r_max_combined)
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

def find_common_spatial_bounds(blue_result, red_result):
    """Find the intersection of valid spatial regions from both datasets."""
    print(f"\n{'='*60}")
    print("FINDING COMMON SPATIAL BOUNDS")
    print(f"{'='*60}")
    
    # get coordinate info from both datasets
    blue_coords = blue_result['coord_info']
    red_coords = red_result['coord_info']
    
    print(f"Blue spatial bounds:")
    print(f"  X: {blue_coords['x_min']:.2f} to {blue_coords['x_max']:.2f} arcsec")
    print(f"  Y: {blue_coords['y_min']:.2f} to {blue_coords['y_max']:.2f} arcsec")
    print(f"  Shape: {blue_result['data'].shape[1:]} (y, x)")
    
    print(f"Red spatial bounds:")
    print(f"  X: {red_coords['x_min']:.2f} to {red_coords['x_max']:.2f} arcsec")
    print(f"  Y: {red_coords['y_min']:.2f} to {red_coords['y_max']:.2f} arcsec")
    print(f"  Shape: {red_result['data'].shape[1:]} (y, x)")
    
    # find intersection of coordinate ranges
    common_x_min = max(blue_coords['x_min'], red_coords['x_min'])
    common_x_max = min(blue_coords['x_max'], red_coords['x_max'])
    common_y_min = max(blue_coords['y_min'], red_coords['y_min'])
    common_y_max = min(blue_coords['y_max'], red_coords['y_max'])
    
    print(f"Common spatial bounds:")
    print(f"  X: {common_x_min:.2f} to {common_x_max:.2f} arcsec")
    print(f"  Y: {common_y_min:.2f} to {common_y_max:.2f} arcsec")
    
    # check if there's valid overlap
    if common_x_max <= common_x_min or common_y_max <= common_y_min:
        raise ValueError("No spatial overlap between blue and red data!")
    
    return {
        'x_min': common_x_min,
        'x_max': common_x_max,
        'y_min': common_y_min,
        'y_max': common_y_max,
        'spatial_sampling': blue_coords['spatial_sampling']  # should be same for both
    }

def crop_to_common_bounds(data_result, common_coords):
    """Crop a dataset to the common spatial bounds."""
    
    # get current coordinates
    current_coords = data_result['coord_info']
    pixel_scale = current_coords['spatial_sampling']
    
    # calculate current grid
    current_height, current_width = data_result['data'].shape[1:]
    current_x_coords = np.linspace(current_coords['x_min'], current_coords['x_max'], current_width)
    current_y_coords = np.linspace(current_coords['y_min'], current_coords['y_max'], current_height)
    
    # find indices for common bounds
    x_mask = (current_x_coords >= common_coords['x_min']) & (current_x_coords <= common_coords['x_max'])
    y_mask = (current_y_coords >= common_coords['y_min']) & (current_y_coords <= common_coords['y_max'])
    
    x_indices = np.where(x_mask)[0]
    y_indices = np.where(y_mask)[0]
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("No pixels found in common bounds!")
    
    x_start, x_end = x_indices[0], x_indices[-1] + 1
    y_start, y_end = y_indices[0], y_indices[-1] + 1
    
    # crop all arrays
    cropped_data = data_result['data'][:, y_start:y_end, x_start:x_end]
    cropped_weights = data_result['weights'][:, y_start:y_end, x_start:x_end] if data_result['weights'] is not None else None
    cropped_variance = data_result['variance'][:, y_start:y_end, x_start:x_end] if data_result['variance'] is not None else None
    
    # update coordinate info to match exactly
    cropped_coords = common_coords.copy()
    
    return {
        'data': cropped_data,
        'weights': cropped_weights,
        'variance': cropped_variance,
        'wavelengths': data_result['wavelengths'],
        'coord_info': cropped_coords,
        'header': data_result['header']
    }

def combine_blue_red_data(blue_result, red_result):
    """Combine blue and red data cubes into a single dataset with matching spatial dimensions."""
    print(f"\n{'='*60}")
    print("COMBINING BLUE AND RED DATA")
    print(f"{'='*60}")
    
    # find common spatial bounds
    common_coords = find_common_spatial_bounds(blue_result, red_result)
    
    # crop both datasets to common bounds
    print("\nCropping blue data to common bounds...")
    blue_cropped = crop_to_common_bounds(blue_result, common_coords)
    
    print("Cropping red data to common bounds...")
    red_cropped = crop_to_common_bounds(red_result, common_coords)
    
    # verify spatial dimensions now match
    blue_shape = blue_cropped['data'].shape
    red_shape = red_cropped['data'].shape
    
    if blue_shape[1:] != red_shape[1:]:
        raise ValueError(f"Spatial dimensions still don't match after cropping: Blue {blue_shape[1:]} vs Red {red_shape[1:]}")
    
    print(f"\nAfter cropping to common bounds:")
    print(f"Blue data shape: {blue_shape}")
    print(f"Red data shape: {red_shape}")
    
    # combine wavelength arrays
    blue_waves = blue_cropped['wavelengths']
    red_waves = red_cropped['wavelengths']
    
    print(f"Blue wavelength range: {blue_waves[0]:.1f} - {blue_waves[-1]:.1f} Å ({len(blue_waves)} bins)")
    print(f"Red wavelength range: {red_waves[0]:.1f} - {red_waves[-1]:.1f} Å ({len(red_waves)} bins)")
    
    # sort combined wavelengths
    combined_waves = np.concatenate([blue_waves, red_waves])
    wave_sort_idx = np.argsort(combined_waves)
    combined_waves = combined_waves[wave_sort_idx]
    
    # combine data cubes
    combined_data = np.concatenate([blue_cropped['data'], red_cropped['data']], axis=0)
    combined_data = combined_data[wave_sort_idx, :, :]
    
    # combine weights if available
    combined_weights = None
    if blue_cropped['weights'] is not None and red_cropped['weights'] is not None:
        combined_weights = np.concatenate([blue_cropped['weights'], red_cropped['weights']], axis=0)
        combined_weights = combined_weights[wave_sort_idx, :, :]
    elif blue_cropped['weights'] is not None:
        # create zero weights for red data
        red_weights_dummy = np.zeros_like(red_cropped['data'])
        combined_weights = np.concatenate([blue_cropped['weights'], red_weights_dummy], axis=0)
        combined_weights = combined_weights[wave_sort_idx, :, :]
    elif red_cropped['weights'] is not None:
        # create zero weights for blue data
        blue_weights_dummy = np.zeros_like(blue_cropped['data'])
        combined_weights = np.concatenate([blue_weights_dummy, red_cropped['weights']], axis=0)
        combined_weights = combined_weights[wave_sort_idx, :, :]
    
    # combine variance if available
    combined_variance = None
    if blue_cropped['variance'] is not None and red_cropped['variance'] is not None:
        combined_variance = np.concatenate([blue_cropped['variance'], red_cropped['variance']], axis=0)
        combined_variance = combined_variance[wave_sort_idx, :, :]
    elif blue_cropped['variance'] is not None:
        # create high variance for red data (indicating no data)
        red_var_dummy = np.full_like(red_cropped['data'], np.inf)
        combined_variance = np.concatenate([blue_cropped['variance'], red_var_dummy], axis=0)
        combined_variance = combined_variance[wave_sort_idx, :, :]
    elif red_cropped['variance'] is not None:
        # create high variance for blue data (indicating no data)
        blue_var_dummy = np.full_like(blue_cropped['data'], np.inf)
        combined_variance = np.concatenate([blue_var_dummy, red_cropped['variance']], axis=0)
        combined_variance = combined_variance[wave_sort_idx, :, :]
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Combined wavelength range: {combined_waves[0]:.1f} - {combined_waves[-1]:.1f} Å ({len(combined_waves)} bins)")
    print(f"Final spatial bounds: {common_coords['x_min']:.2f} to {common_coords['x_max']:.2f} (X), {common_coords['y_min']:.2f} to {common_coords['y_max']:.2f} (Y)")
    
    return {
        'data': combined_data,
        'weights': combined_weights,
        'variance': combined_variance,
        'wavelengths': combined_waves,
        'coord_info': common_coords,
        'blue_header': blue_cropped['header'],
        'red_header': red_cropped['header'],
        'blue_cropped': blue_cropped,  # keep for plotting
        'red_cropped': red_cropped     # keep for plotting
    }

def extract_wavelength_indices_with_resolution(windows, wavelengths, blue_waves, red_waves, blue_resolution, red_resolution):
    """Convert wavelength ranges to bin indices and calculate resolution for each window."""
    
    print(f"\nMapping windows to spectral bins with resolution info:")
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
        
        # determine which instrument(s) contribute to this window
        window_waves = wavelengths[start_idx:end_idx+1]
        blue_contribution = np.sum((window_waves >= blue_waves[0]) & (window_waves <= blue_waves[-1]))
        red_contribution = np.sum((window_waves >= red_waves[0]) & (window_waves <= red_waves[-1]))
        
        # calculate effective resolution
        if blue_contribution > 0 and red_contribution > 0:
            # mixed window - use weighted average resolution
            blue_weight = blue_contribution / window['n_bins']
            red_weight = red_contribution / window['n_bins']
            window['resolution'] = blue_weight * blue_resolution + red_weight * red_resolution
            window['instrument'] = f"Blue+Red (Blue: {blue_contribution}, Red: {red_contribution})"
        elif blue_contribution > 0:
            window['resolution'] = blue_resolution
            window['instrument'] = f"Blue ({blue_contribution} bins)"
        elif red_contribution > 0:
            window['resolution'] = red_resolution
            window['instrument'] = f"Red ({red_contribution} bins)"
        else:
            window['resolution'] = (blue_resolution + red_resolution) / 2  # fallback
            window['instrument'] = "Unknown"
        
        print(f"  {window['name']}:")
        print(f"    Requested: {window['r_min']:.1f} - {window['r_max']:.1f} Å")
        print(f"    Actual: {window['actual_r_min']:.1f} - {window['actual_r_max']:.1f} Å")
        print(f"    Bins: {start_idx} - {end_idx} ({window['n_bins']} bins)")
        print(f"    Resolution: {window['resolution']:.0f}")
        print(f"    Instrument: {window['instrument']}")
    
    return windows

def write_metadata(coord_info, windows, output_dir, ni, nj):
    """Write updated metadata file in the new format."""
    
    output_dir = Path(output_dir)
    metadata_file = output_dir / 'metadata.txt'
    
    print(f"  Metadata: {metadata_file}")
    
    with open(metadata_file, 'w') as f:
        f.write("# Blobby3D Metadata File - Combined Blue+Red Multi-Window Configuration\n")
        f.write("# Generated by Combined SAMI Blue-Red IFS Processor\n\n")
        
        # spatial dimensions
        f.write(f"ni {ni}\n")
        f.write(f"nj {nj}\n")
        f.write(f"x_min {coord_info['x_min']:.6f}\n")
        f.write(f"x_max {coord_info['x_max']:.6f}\n")
        f.write(f"y_min {coord_info['y_min']:.6f}\n")
        f.write(f"y_max {coord_info['y_max']:.6f}\n")
        
        # wavelength windows in new format
        for i, window in enumerate(windows):
            # calculate spectral resolution from resolution number and central wavelength
            central_wavelength = (window['actual_r_min'] + window['actual_r_max']) / 2.0
            spectral_resolution = window['resolution']
            
            f.write(f"wave_range {window['actual_r_min']:.6f} {window['actual_r_max']:.6f} "
                   f"{window['new_start_bin']} {window['new_end_bin']} {window['n_bins']} "
                   f"{spectral_resolution:.0f}  # {window['name']} ({window['instrument']})\n")
        
        f.write(f"\n# Total wavelength bins: {sum(w['n_bins'] for w in windows)}\n")
        f.write(f"# Spectral resolution format: R = λ/Δλ where λ is central wavelength\n")
        f.write(f"# Instrument contributions shown in comments\n")
    
    return metadata_file

def extract_windowed_data(data_cube, var_cube, windows):
    """Extract data for the specified wavelength windows."""
    
    nr_orig, ni, nj = data_cube.shape
    
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
        print(f"data cub shape: {np.shape(data_cube)}")
        data_cube_transposed = data_cube[start_idx:end_idx+1, :, :].transpose(1, 2, 0)
        windowed_data[:, :, current_bin:current_bin+n_bins] = data_cube_transposed
        if var_cube is not None:
            var_cube_transposed = var_cube[start_idx:end_idx+1, :, :].transpose(1, 2, 0)
            windowed_var[:, :, current_bin:current_bin+n_bins] = var_cube_transposed
        
        # update window with position in new array
        window['new_start_bin'] = current_bin
        window['new_end_bin'] = current_bin + n_bins - 1
        
        current_bin += n_bins
    
    print(f"\nExtracted windowed data:")
    print(f"  Original shape: {data_cube.shape}")
    print(f"  Windowed shape: {windowed_data.shape}")
    print(f"  Data reduction: {nr_orig} → {total_bins} bins ({100*total_bins/nr_orig:.1f}%)")
    
    return windowed_data, windowed_var

def write_windowed_data(windowed_data, windowed_weights, windowed_var, output_dir):
    """Write windowed data to text files."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # flatten spatial dimensions via reshape
    data_flat = windowed_data.reshape(-1, windowed_data.shape[-1])
    var_flat = windowed_var.reshape(-1, windowed_var.shape[-1]) if windowed_var is not None else None
    weights_flat = windowed_weights.reshape(-1, windowed_weights.shape[-1]) if windowed_weights is not None else None
    
    data_file = output_dir / 'data.txt'
    var_file = output_dir / 'var.txt'
    weights_file = output_dir / 'weights.txt'
    
    print(f"\nWriting windowed data:")
    print(f"  Data: {data_file}")
    
    np.savetxt(data_file, data_flat, fmt='%.8e')
    
    if var_flat is not None:
        print(f"  Variance: {var_file}")
        np.savetxt(var_file, var_flat, fmt='%.8e')
    
    if weights_flat is not None:
        print(f"  Weights: {weights_file}")
        np.savetxt(weights_file, weights_flat, fmt='%.8e')
    
    return data_file, var_file, weights_file

def write_metadata(coord_info, windows, output_dir, ni, nj):
    """Write updated metadata file in the new format."""
    
    output_dir = Path(output_dir)
    metadata_file = output_dir / 'metadata.txt'
    
    print(f"  Metadata: {metadata_file}")
    
    with open(metadata_file, 'w') as f:
        f.write("# Blobby3D Metadata File - Combined Blue+Red Multi-Window Configuration\n")
        f.write("# Generated by Combined SAMI Blue-Red IFS Processor\n\n")
        
        # spatial dimensions
        f.write(f"ni {ni}\n")
        f.write(f"nj {nj}\n")
        f.write(f"x_min {coord_info['x_min']:.6f}\n")
        f.write(f"x_max {coord_info['x_max']:.6f}\n")
        f.write(f"y_min {coord_info['y_min']:.6f}\n")
        f.write(f"y_max {coord_info['y_max']:.6f}\n")
        
        # wavelength windows in new format
        for i, window in enumerate(windows):
            f.write(f"wave_range {window['actual_r_min']:.6f} {window['actual_r_max']:.6f} "
                   f"{window['new_start_bin']} {window['new_end_bin']} {window['n_bins']} "
                   f"{window['resolution']:.0f}  # {window['name']} ({window['instrument']})\n")
        
        f.write(f"\n# Total wavelength bins: {sum(w['n_bins'] for w in windows)}\n")
        f.write(f"# Resolution format: {window['resolution']:.0f} indicates spectral resolution\n")
        f.write(f"# Instrument contributions shown in comments\n")
    
    return metadata_file

def create_comparison_plot(blue_result, red_result, combined_result, windowed_data, windows, output_dir):
    """Create comparison plots showing blue, red, combined, and windowed data."""
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    
    # get spatial extents
    coord_info = combined_result['coord_info']
    x_extent = [coord_info['x_min'], coord_info['x_max']]
    y_extent = [coord_info['y_min'], coord_info['y_max']]
    
    # use the cropped versions for plotting to ensure same dimensions
    blue_cropped = combined_result['blue_cropped']
    red_cropped = combined_result['red_cropped']
    
    # wavelength arrays
    blue_waves = blue_cropped['wavelengths']
    red_waves = red_cropped['wavelengths']
    combined_waves = combined_result['wavelengths']
    
    # windowed wavelengths
    windowed_waves = []
    for window in windows:
        start_idx = window['start_idx']
        end_idx = window['end_idx']
        window_waves = combined_waves[start_idx:end_idx+1]
        windowed_waves.extend(window_waves)
    windowed_waves = np.array(windowed_waves)
    
    # 1. flux maps for each dataset
    blue_flux = np.nansum(blue_cropped['data'], axis=0)
    red_flux = np.nansum(red_cropped['data'], axis=0)
    combined_flux = np.nansum(combined_result['data'], axis=0)
    windowed_flux = np.nansum(windowed_data, axis=2)
    
    im1 = axes[0,0].imshow(blue_flux, origin='lower', aspect='auto', extent=x_extent + y_extent)
    axes[0,0].set_title('Blue Total Flux Map (Cropped)')
    axes[0,0].set_xlabel('X (arcsec)')
    axes[0,0].set_ylabel('Y (arcsec)')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(red_flux, origin='lower', aspect='auto', extent=x_extent + y_extent)
    axes[0,1].set_title('Red Total Flux Map (Cropped)')
    axes[0,1].set_xlabel('X (arcsec)')
    axes[0,1].set_ylabel('Y (arcsec)')
    plt.colorbar(im2, ax=axes[0,1])
    
    im3 = axes[1,0].imshow(combined_flux, origin='lower', aspect='auto', extent=x_extent + y_extent)
    axes[1,0].set_title('Combined Total Flux Map')
    axes[1,0].set_xlabel('X (arcsec)')
    axes[1,0].set_ylabel('Y (arcsec)')
    plt.colorbar(im3, ax=axes[1,0])
    
    im4 = axes[1,1].imshow(windowed_flux, origin='lower', aspect='auto', extent=x_extent + y_extent)
    axes[1,1].set_title('Windowed Total Flux Map')
    axes[1,1].set_xlabel('X (arcsec)')
    axes[1,1].set_ylabel('Y (arcsec)')
    plt.colorbar(im4, ax=axes[1,1])
    
    # 2. central spectra
    ni, nj = blue_cropped['data'].shape[1], blue_cropped['data'].shape[2]
    center_i, center_j = ni//2, nj//2
    
    blue_spectrum = blue_cropped['data'][:, center_i, center_j]
    red_spectrum = red_cropped['data'][:, center_i, center_j]
    combined_spectrum = combined_result['data'][:, center_i, center_j]
    windowed_spectrum = windowed_data[center_i, center_j, :]
    
    # plot blue and red separately
    axes[2,0].plot(blue_waves, blue_spectrum, 'b-', linewidth=1, label='Blue')
    axes[2,0].set_title(f'Blue Central Spectrum (Cropped)\n(spaxel {center_i}, {center_j})')
    axes[2,0].set_xlabel('Wavelength (Å)')
    axes[2,0].set_ylabel('Flux')
    axes[2,0].grid(True, alpha=0.3)
    
    axes[2,1].plot(red_waves, red_spectrum, 'r-', linewidth=1, label='Red')
    axes[2,1].set_title(f'Red Central Spectrum (Cropped)\n(spaxel {center_i}, {center_j})')
    axes[2,1].set_xlabel('Wavelength (Å)')
    axes[2,1].set_ylabel('Flux')
    axes[2,1].grid(True, alpha=0.3)
    
    # plot combined with window overlays
    axes[3,0].plot(combined_waves, combined_spectrum, 'k-', linewidth=1, label='Combined')
    axes[3,0].set_title(f'Combined Central Spectrum\n(spaxel {center_i}, {center_j})')
    axes[3,0].set_xlabel('Wavelength (Å)')
    axes[3,0].set_ylabel('Flux')
    axes[3,0].grid(True, alpha=0.3)
    
    # add window regions
    colors = plt.cm.Set3(np.linspace(0, 1, len(windows)))
    for i, window in enumerate(windows):
        axes[3,0].axvspan(window['actual_r_min'], window['actual_r_max'], 
                         alpha=0.3, color=colors[i], label=window['name'])
    
    # only show legend if we have few windows
    if len(windows) <= 5:
        axes[3,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # plot windowed spectrum
    axes[3,1].plot(windowed_waves, windowed_spectrum, 'g-', linewidth=1, label='Windowed')
    axes[3,1].scatter(windowed_waves, windowed_spectrum, s=10, alpha=0.7)
    axes[3,1].set_title(f'Windowed Central Spectrum\n(spaxel {center_i}, {center_j})')
    axes[3,1].set_xlabel('Wavelength (Å)')
    axes[3,1].set_ylabel('Flux')
    axes[3,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save plot
    plot_file = Path(output_dir) / 'combined_windowed_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Comparison plot: {plot_file}")
    
    return fig

def print_summary(blue_result, red_result, combined_result, windows, windowed_data):
    """Print summary of the processing."""
    
    print("\n" + "="*80)
    print("COMBINED BLUE-RED WINDOWING SUMMARY")
    print("="*80)
    
    blue_shape = blue_result['data'].shape
    red_shape = red_result['data'].shape
    combined_shape = combined_result['data'].shape
    windowed_shape = windowed_data.shape
    
    print(f"Blue data shape: {blue_shape}")
    print(f"Red data shape: {red_shape}")
    print(f"Combined data shape: {combined_shape}")
    print(f"Windowed data shape: {windowed_shape}")
    
    print(f"\nData reduction:")
    print(f"  Blue: {blue_shape[0]} bins")
    print(f"  Red: {red_shape[0]} bins")
    print(f"  Combined: {combined_shape[0]} bins")
    print(f"  Windowed: {windowed_shape[2]} bins ({100*windowed_shape[2]/combined_shape[0]:.1f}% of combined)")
    
    coord_info = combined_result['coord_info']
    print(f"\nSpatial coverage:")
    print(f"  {windowed_shape[0]} × {windowed_shape[1]} spaxels")
    print(f"  {coord_info['x_min']:.1f}″ to {coord_info['x_max']:.1f}″ (X)")
    print(f"  {coord_info['y_min']:.1f}″ to {coord_info['y_max']:.1f}″ (Y)")
    
    blue_waves = blue_result['wavelengths']
    red_waves = red_result['wavelengths']
    combined_waves = combined_result['wavelengths']
    
    print(f"\nWavelength coverage:")
    print(f"  Blue: {blue_waves[0]:.1f} - {blue_waves[-1]:.1f} Å")
    print(f"  Red: {red_waves[0]:.1f} - {red_waves[-1]:.1f} Å")
    print(f"  Combined: {combined_waves[0]:.1f} - {combined_waves[-1]:.1f} Å")
    
    if windows:
        total_coverage = sum(w['actual_width'] for w in windows)
        print(f"  Windowed: {total_coverage:.1f} Å total ({len(windows)} windows)")
    
    print(f"\nWindow details:")
    for i, window in enumerate(windows):
        print(f"  Window {i+1}: {window['name']}")
        print(f"    {window['actual_r_min']:.1f} - {window['actual_r_max']:.1f} Å ({window['n_bins']} bins)")
        print(f"    Resolution: {window['resolution']:.0f}")
        print(f"    Instrument: {window['instrument']}")
        print(f"    Position: bins {window['new_start_bin']} - {window['new_end_bin']}")
    
    print("="*80)

def main():
    """Main function for processing combined SAMI blue and red IFS data."""
    
    # +++++++++++++ Can be Edited +++++++++++++++++
    
    # input files
    input_file_blue = '/Users/scol0322/Documents/sami_data/65410_blue_7_Y13SAR1_P003_15T008_2013_03_05-2013_03_17.fits'
    blue_data_extension = 'PRIMARY'
    input_file_red = '/Users/scol0322/Documents/sami_data/65410_red_7_Y13SAR1_P003_15T008_2013_03_05-2013_03_17.fits'
    red_data_extension = 'PRIMARY'
    
    # continuum file
    continuum_file = '/Users/scol0322/Documents/sami_data/65410_7_Y13SAR1_P003_15T008_2013_03_05-2013_03_17_1_comp.fits'
    blue_extension_name = 'B_CONTINUUM'
    red_extension_name = 'R_CONTINUUM'

    # instrument resolutions
    blue_resolution = 1812
    red_resolution = 4326
    
    # output directory
    output_dir = '/Users/scol0322/Documents/output/65410-combined'
    
    # processing parameters
    pix_size_arcsec = 0.5              # SAMI spatial sampling
    redshift = 0.043514                # redshift for correction (z)
    remove_invalid_spaxels = True      # remove spaxels with all NaN
    
    # windowing parameters
    window_size = 15.0                 # size (radius; angstrom) of each window
    min_gap = 5.0                      # minimum gap before combining windows (angstrom)
    
    # plotting
    create_plots = True                # set to False to skip plotting
    
    # common emission lines (wavelength in angstroms)
    emission_lines = {
        '[OII]3726': 3726.03,    # in blue image
        '[OII]3729': 3728.82,    # in blue image
        'H-beta': 4861.3,        # in blue image
        '[OIII]5007': 5006.8,    # in blue image
        'H-alpha': 6562.81,      # red image
        '[NII]6548': 6548.1,     # red image
        '[NII]6583': 6583.1,     # red image
        '[SII]6717': 6716.4,     # red image
        '[SII]6731': 6730.8      # red image
    }
    
    # +++++++++ End of Can be Edited +++++++++++++
    
    # check input files exist
    for file in [input_file_blue, input_file_red]:
        if not Path(file).exists():
            print(f"Error: Input file {file} not found")
            sys.exit(1)
    
    # create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Combined SAMI Blue-Red IFS Data Processor for Blobby3D")
    print("=" * 60)
    print(f"Blue input file: {input_file_blue}")
    print(f"Red input file: {input_file_red}")
    print(f"Output directory: {output_dir}")
    print(f"Blue resolution: {blue_resolution}")
    print(f"Red resolution: {red_resolution}")
    print(f"Redshift: {redshift}")
    print(f"Window size: ±{window_size} Å")
    print(f"Minimum gap for combining: {min_gap} Å")
    print()
    
    try:
        # process blue and red data separately
        blue_result = process_single_fits(input_file_blue, pix_size_arcsec, redshift, remove_invalid_spaxels, continuum_file, blue_extension_name)
        red_result = process_single_fits(input_file_red, pix_size_arcsec, redshift, remove_invalid_spaxels, continuum_file, red_extension_name)
        
        # combine blue and red data
        combined_result = combine_blue_red_data(blue_result, red_result)
        
        # set all nan to zero in all data
        combined_result['data'] = np.nan_to_num(combined_result['data'], nan=0.0)
        if combined_result['weights'] is not None:
            combined_result['weights'] = np.nan_to_num(combined_result['weights'], nan=0.0)
        if combined_result['variance'] is not None:
            combined_result['variance'] = np.nan_to_num(combined_result['variance'], nan=0.0, posinf=1e20)

        # define emission line windows
        print(f"\nDefining wavelength windows (±{window_size} Å around emission lines):")
        windows = define_emission_line_windows(emission_lines, window_size)
        
        # filter windows to those within the combined data range
        windows = filter_windows_by_coverage(windows, combined_result['wavelengths'])
        
        if not windows:
            print("No emission lines found within the combined wavelength range!")
            sys.exit(1)
        
        # combine overlapping windows
        windows = combine_overlapping_windows(windows, min_gap)
        
        # convert to bin indices with resolution information
        windows = extract_wavelength_indices_with_resolution(
            windows, combined_result['wavelengths'], 
            blue_result['wavelengths'], red_result['wavelengths'],
            blue_resolution, red_resolution)
        
        # extract windowed data
        windowed_data, windowed_var = extract_windowed_data(
            combined_result['data'], combined_result['variance'], windows)
        
        # save windowed data and metadata
        write_windowed_data(windowed_data, combined_result['weights'], windowed_var, output_dir)
        write_metadata(combined_result['coord_info'], windows, output_dir, 
                      windowed_data.shape[0], windowed_data.shape[1])
        
        # create comparison plot if requested
        if create_plots:
            print(f"\nCreating comparison plots:")
            create_comparison_plot(blue_result, red_result, combined_result, 
                                 windowed_data, windows, output_dir)
        
        # print summary
        print_summary(blue_result, red_result, combined_result, windows, windowed_data)
        
        print(f"\nProcessing complete! Combined windowed data saved to: {output_dir}")
        print(f"Files created:")
        print(f"  - data.txt (windowed flux data)")
        print(f"  - var.txt (windowed variance data)")
        if combined_result['weights'] is not None:
            print(f"  - weights.txt (windowed weights data)")
        print(f"  - metadata.txt (Blobby3D metadata with resolution info)")
        if create_plots:
            print(f"  - combined_windowed_comparison.png (comparison plots)")
        
    except Exception as e:
        print(f"Error processing SAMI data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()