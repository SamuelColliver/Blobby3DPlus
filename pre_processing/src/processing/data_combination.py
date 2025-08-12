#!/usr/bin/env python3
"""
Functions for combining multiple IFS datasets (e.g., blue + red arms).
"""

import numpy as np


def find_common_spatial_bounds(*data_results):
    """
    Find the intersection of valid spatial regions from multiple datasets.
    
    Parameters
    ----------
    *data_results : dict
        Variable number of data result dictionaries, each containing 'coord_info'
    
    Returns
    -------
    dict
        Common coordinate information
    """
    print(f"\n{'='*60}")
    print("FINDING COMMON SPATIAL BOUNDS")
    print(f"{'='*60}")
    
    if len(data_results) < 2:
        raise ValueError("Need at least 2 datasets to find common bounds")
    
    # get coordinate info from all datasets
    coord_infos = [result['coord_info'] for result in data_results]
    
    for i, coords in enumerate(coord_infos):
        print(f"Dataset {i+1} spatial bounds:")
        print(f"  X: {coords['x_min']:.2f} to {coords['x_max']:.2f} arcsec")
        print(f"  Y: {coords['y_min']:.2f} to {coords['y_max']:.2f} arcsec")
        print(f"  Shape: {data_results[i]['data'].shape[1:]} (y, x)")
    
    # find intersection of coordinate ranges
    common_x_min = max(coords['x_min'] for coords in coord_infos)
    common_x_max = min(coords['x_max'] for coords in coord_infos)
    common_y_min = max(coords['y_min'] for coords in coord_infos)
    common_y_max = min(coords['y_max'] for coords in coord_infos)
    
    print(f"Common spatial bounds:")
    print(f"  X: {common_x_min:.2f} to {common_x_max:.2f} arcsec")
    print(f"  Y: {common_y_min:.2f} to {common_y_max:.2f} arcsec")
    
    # check if there's valid overlap
    if common_x_max <= common_x_min or common_y_max <= common_y_min:
        raise ValueError("No spatial overlap between datasets!")
    
    # use the spatial sampling from the first dataset (should be same for all)
    spatial_sampling = coord_infos[0]['spatial_sampling']
    
    return {
        'x_min': common_x_min,
        'x_max': common_x_max,
        'y_min': common_y_min,
        'y_max': common_y_max,
        'spatial_sampling': spatial_sampling
    }


def crop_to_common_bounds(data_result, common_coords):
    """
    Crop a dataset to the common spatial bounds.
    
    Parameters
    ----------
    data_result : dict
        Data result dictionary with 'data', 'coord_info', etc.
    common_coords : dict
        Common coordinate bounds
    
    Returns
    -------
    dict
        Cropped data result
    """
    # get current coordinates
    current_coords = data_result['coord_info']
    
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
    
    return {
        'data': cropped_data,
        'weights': cropped_weights,
        'variance': cropped_variance,
        'wavelengths': data_result['wavelengths'],
        'coord_info': common_coords.copy(),
        'header': data_result['header']
    }


def combine_spectral_data(*data_results, sort_wavelengths=True):
    """
    Combine multiple spectral datasets into a single dataset.
    
    Parameters
    ----------
    *data_results : dict
        Variable number of cropped data result dictionaries
    sort_wavelengths : bool
        Whether to sort the combined wavelengths
    
    Returns
    -------
    dict
        Combined data result
    """
    print(f"\n{'='*60}")
    print(f"COMBINING {len(data_results)} SPECTRAL DATASETS")
    print(f"{'='*60}")
    
    if len(data_results) < 2:
        raise ValueError("Need at least 2 datasets to combine")
    
    # verify spatial dimensions match
    ref_shape = data_results[0]['data'].shape[1:]
    for i, result in enumerate(data_results[1:], 1):
        if result['data'].shape[1:] != ref_shape:
            raise ValueError(f"Spatial dimensions don't match: Dataset 1 {ref_shape} vs Dataset {i+1} {result['data'].shape[1:]}")
    
    # combine wavelength arrays
    all_wavelengths = []
    all_data = []
    all_weights = []
    all_variance = []
    
    for i, result in enumerate(data_results):
        waves = result['wavelengths']
        print(f"Dataset {i+1} wavelength range: {waves[0]:.1f} - {waves[-1]:.1f} Å ({len(waves)} bins)")
        
        all_wavelengths.append(waves)
        all_data.append(result['data'])
        
        if result['weights'] is not None:
            all_weights.append(result['weights'])
        else:
            # create zero weights for missing data
            all_weights.append(np.zeros_like(result['data']))
        
        if result['variance'] is not None:
            all_variance.append(result['variance'])
        else:
            # create high variance for missing data
            all_variance.append(np.full_like(result['data'], np.inf))
    
    # concatenate arrays
    combined_waves = np.concatenate(all_wavelengths)
    combined_data = np.concatenate(all_data, axis=0)
    combined_weights = np.concatenate(all_weights, axis=0)
    combined_variance = np.concatenate(all_variance, axis=0)
    
    # sort by wavelength if requested
    if sort_wavelengths:
        wave_sort_idx = np.argsort(combined_waves)
        combined_waves = combined_waves[wave_sort_idx]
        combined_data = combined_data[wave_sort_idx, :, :]
        combined_weights = combined_weights[wave_sort_idx, :, :]
        combined_variance = combined_variance[wave_sort_idx, :, :]
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Combined wavelength range: {combined_waves[0]:.1f} - {combined_waves[-1]:.1f} Å ({len(combined_waves)} bins)")
    
    # use coordinate info from first dataset (should be same after cropping)
    coord_info = data_results[0]['coord_info']
    print(f"Final spatial bounds: {coord_info['x_min']:.2f} to {coord_info['x_max']:.2f} (X), {coord_info['y_min']:.2f} to {coord_info['y_max']:.2f} (Y)")
    
    # collect headers
    headers = {f'header_{i+1}': result['header'] for i, result in enumerate(data_results)}
    
    combined_result = {
        'data': combined_data,
        'weights': combined_weights,
        'variance': combined_variance,
        'wavelengths': combined_waves,
        'coord_info': coord_info,
        **headers,
        'individual_results': data_results  # keep for reference
    }
    
    return combined_result


def combine_multi_arm_data(*data_results):
    """
    High-level function to combine multiple IFS arm datasets.
    
    This function handles the full workflow:
    1. Find common spatial bounds
    2. Crop all datasets to common bounds  
    3. Combine into single dataset
    
    Parameters
    ----------
    *data_results : dict
        Variable number of data result dictionaries
    
    Returns
    -------
    dict
        Combined data result
    """
    # find common spatial bounds
    common_coords = find_common_spatial_bounds(*data_results)
    
    # crop all datasets to common bounds
    cropped_results = []
    for i, result in enumerate(data_results):
        print(f"\nCropping dataset {i+1} to common bounds...")
        cropped = crop_to_common_bounds(result, common_coords)
        cropped_results.append(cropped)
    
    # combine into single dataset
    combined_result = combine_spectral_data(*cropped_results)
    
    return combined_result