#!/usr/bin/env python3
"""
Spectral processing functions for IFS data.
"""

import numpy as np
import warnings
from pathlib import Path


def redshift_correct_wavelengths(wavelengths, redshift):
    """
    Apply redshift correction to wavelengths.
    
    Parameters
    ----------
    wavelengths : np.ndarray
        Observed wavelengths in Angstroms
    redshift : float
        Redshift value (z)
    
    Returns
    -------
    np.ndarray
        Rest-frame wavelengths in Angstroms
    """
    if redshift == 0:
        return wavelengths
    
    rest_wavelengths = wavelengths / (1 + redshift)
    print(f"Applied redshift correction (z={redshift:.6f})")
    print(f"Rest-frame wavelength range: {rest_wavelengths[0]:.2f} - {rest_wavelengths[-1]:.2f} Å")
    
    return rest_wavelengths


def subtract_continuum_nanmedian(data, weights=None):
    """
    Subtract continuum using nanmedian across wavelength axis.
    
    Parameters
    ----------
    data : np.ndarray
        Data cube in (wavelength, y, x) format
    weights : np.ndarray, optional
        Weights array (not modified by continuum subtraction)
    
    Returns
    -------
    np.ndarray
        Continuum-subtracted data
    """
    print("Subtracting continuum using nanmedian across wavelength axis...")
    
    # calculate continuum (median along wavelength axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        continuum = np.nanmedian(data, axis=0, keepdims=True)
    
    # replace NaN continuum values with 0 to avoid issues
    continuum = np.nan_to_num(continuum, nan=0.0)
    
    # subtract continuum from data
    continuum_subtracted = data - continuum
    
    print(f"Continuum range: {np.nanmin(continuum):.2e} to {np.nanmax(continuum):.2e}")
    print(f"Data range after continuum subtraction: {np.nanmin(continuum_subtracted):.2e} to {np.nanmax(continuum_subtracted):.2e}")
    
    return continuum_subtracted


def subtract_continuum_from_file(data, continuum, weights=None):
    """
    Subtract continuum loaded from external file.
    
    Parameters
    ----------
    data : np.ndarray
        Data cube in (wavelength, y, x) format
    continuum : np.ndarray
        Continuum cube in same format as data
    weights : np.ndarray, optional
        Weights array (not modified)
    
    Returns
    -------
    np.ndarray
        Continuum-subtracted data
    """
    print("Subtracting continuum from external file...")
    
    if data.shape != continuum.shape:
        print(f'ERROR: data shape {data.shape} does not match continuum {continuum.shape}')
        print("Falling back to nanmedian subtraction...")
        return subtract_continuum_nanmedian(data, weights)
    
    continuum_subtracted = data - continuum
    
    print(f"Continuum range: {np.nanmin(continuum):.2e} to {np.nanmax(continuum):.2e}")
    print(f"Data range after continuum subtraction: {np.nanmin(continuum_subtracted):.2e} to {np.nanmax(continuum_subtracted):.2e}")
    
    return continuum_subtracted


def clean_nan_edges(data, pixelscale_arcsec, weights=None, variance=None):
    """
    Remove rows and columns from edges where all wavelengths are NaN.
    
    Parameters
    ----------
    data : np.ndarray
        Data cube in (wavelength, y, x) format
    pixelscale_arcsec : float
        Pixel scale in arcseconds per pixel
    weights : np.ndarray, optional
        Weights array to crop consistently
    variance : np.ndarray, optional
        Variance array to crop consistently
    
    Returns
    -------
    tuple
        (cropped_data, cropped_weights, cropped_variance, valid_mask, coord_info)
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
    new_x_left = original_x_range[0] + (cols_removed_left * pixelscale_arcsec)
    new_x_right = original_x_range[1] - (cols_removed_right * pixelscale_arcsec)
    new_y_bottom = original_y_range[0] + (rows_removed_bottom * pixelscale_arcsec)
    new_y_top = original_y_range[1] - (rows_removed_top * pixelscale_arcsec)
    
    # crop the data
    if data.ndim == 3:
        cropped_data = data[:, row_start:row_end, col_start:col_end]
        cropped_weights = weights[:, row_start:row_end, col_start:col_end] if weights is not None else None
        cropped_variance = variance[:, row_start:row_end, col_start:col_end] if variance is not None else None
        valid_mask = ~np.all(np.isnan(cropped_data), axis=0)
    else:
        cropped_data = data[row_start:row_end, col_start:col_end]
        cropped_weights = weights[row_start:row_end, col_start:col_end] if weights is not None else None
        cropped_variance = variance[row_start:row_end, col_start:col_end] if variance is not None else None
        valid_mask = ~np.isnan(cropped_data)
    
    # create coordinate info dictionary
    coord_info = {
        'x_min': new_x_left,
        'x_max': new_x_right,
        'y_min': new_y_bottom,
        'y_max': new_y_top,
        'spatial_sampling': pixelscale_arcsec
    }
    
    # calculate statistics
    original_size = original_height * original_width
    cropped_size = cropped_data.shape[-2] * cropped_data.shape[-1]
    removed_count = original_size - cropped_size
    
    print(f"Original dimensions: {data.shape}")
    print(f"Cropped dimensions: {cropped_data.shape}")
    print(f"Removed {removed_count} edge spaxels out of {original_size} total")
    print(f"Coordinate ranges: X=[{new_x_left:.2f}, {new_x_right:.2f}], Y=[{new_y_bottom:.2f}, {new_y_top:.2f}] arcsec")
    
    return cropped_data, cropped_weights, cropped_variance, valid_mask, coord_info


def clean_invalid_values(data, weights=None, variance=None, 
                        nan_replacement=0.0, inf_replacement=1e20):
    """
    Replace NaN and infinite values in data arrays.
    
    Parameters
    ----------
    data : np.ndarray
        Data array to clean
    weights : np.ndarray, optional
        Weights array to clean
    variance : np.ndarray, optional
        Variance array to clean
    nan_replacement : float
        Value to replace NaN with
    inf_replacement : float
        Value to replace infinity with
    
    Returns
    -------
    tuple
        (cleaned_data, cleaned_weights, cleaned_variance)
    """
    print("Cleaning invalid values (NaN/inf)...")
    
    # clean data
    cleaned_data = np.nan_to_num(data, nan=nan_replacement, posinf=inf_replacement, neginf=-inf_replacement)
    
    # clean weights
    cleaned_weights = None
    if weights is not None:
        cleaned_weights = np.nan_to_num(weights, nan=0.0, posinf=inf_replacement, neginf=0.0)
    
    # clean variance  
    cleaned_variance = None
    if variance is not None:
        cleaned_variance = np.nan_to_num(variance, nan=inf_replacement, posinf=inf_replacement, neginf=inf_replacement)
    
    print(f"Replaced NaN with {nan_replacement}, inf with ±{inf_replacement}")
    
    return cleaned_data, cleaned_weights, cleaned_variance