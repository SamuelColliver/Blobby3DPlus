#!/usr/bin/env python3
"""
IFS Data Processing Pipeline for Blobby3D

A modular pipeline for processing Integral Field Spectroscopy (IFS) data
from various instruments into Blobby3D format with emission line windowing.
"""

__version__ = "1.0.0"
__author__ = "IFS Processing Team"

# high-level workflow functions
from .core.processing_workflow import (
    process_windowed_ifs_data,
    process_existing_blobby_data,
    process_single_ifs_file
)

# io functions
from .io.fits_readers import read_sami_fits, read_generic_ifs_fits, read_continuum_fits
from .io.blobby_io import (
    load_blobby_metadata, 
    load_blobby_data_cube, 
    write_blobby_data, 
    write_blobby_metadata,
    convert_old_to_new_metadata
)

# processing functions
from .processing.spectral_processing import (
    redshift_correct_wavelengths,
    subtract_continuum_nanmedian,
    subtract_continuum_from_file,
    clean_nan_edges,
    clean_invalid_values
)

from .processing.windowing import (
    define_emission_line_windows,
    filter_windows_by_coverage,
    combine_overlapping_windows,
    extract_wavelength_indices,
    extract_windowed_data,
    COMMON_EMISSION_LINES
)

from .processing.data_combination import (
    find_common_spatial_bounds,
    crop_to_common_bounds,
    combine_spectral_data,
    combine_multi_arm_data
)

# visualization functions
from .visualization.plotting import (
    create_comparison_plot,
    create_multi_arm_comparison_plot,
    create_processing_summary_plot
)

__all__ = [
    # workflow functions
    'process_windowed_ifs_data',
    'process_existing_blobby_data', 
    'process_single_ifs_file',
    
    # io functions
    'read_sami_fits',
    'read_generic_ifs_fits',
    'read_continuum_fits',
    'load_blobby_metadata',
    'load_blobby_data_cube',
    'write_blobby_data',
    'write_blobby_metadata',
    'convert_old_to_new_metadata',
    
    # spectral processing
    'redshift_correct_wavelengths',
    'subtract_continuum_nanmedian',
    'subtract_continuum_from_file',
    'clean_nan_edges',
    'clean_invalid_values',
    
    # windowing
    'define_emission_line_windows',
    'filter_windows_by_coverage',
    'combine_overlapping_windows', 
    'extract_wavelength_indices',
    'extract_windowed_data',
    'COMMON_EMISSION_LINES',
    
    # data combination
    'find_common_spatial_bounds',
    'crop_to_common_bounds',
    'combine_spectral_data',
    'combine_multi_arm_data',
    
    # visualization
    'create_comparison_plot',
    'create_multi_arm_comparison_plot',
    'create_processing_summary_plot'
]