#!/usr/bin/env python3
"""
High-level processing workflow functions that orchestrate the complete pipeline.
"""

import numpy as np
from pathlib import Path

from ..io.fits_readers import read_sami_fits, read_continuum_fits, read_generic_ifs_fits
from ..processing.spectral_processing import (
    redshift_correct_wavelengths, subtract_continuum_nanmedian, 
    subtract_continuum_from_file, clean_nan_edges, clean_invalid_values
)
from ..processing.windowing import (
    define_emission_line_windows, filter_windows_by_coverage,
    combine_overlapping_windows, extract_wavelength_indices, 
    extract_windowed_data, COMMON_EMISSION_LINES
)
from ..processing.data_combination import combine_multi_arm_data
from ..io.blobby_io import write_blobby_data, write_blobby_metadata
from ..visualization.plotting import (
    create_comparison_plot, create_multi_arm_comparison_plot, 
    create_processing_summary_plot
)


def process_single_ifs_file(
    input_file, 
    pixelscale_arcsec=0.5,
    redshift=0.0,
    instrument='sami',
    continuum_file=None,
    continuum_extension=None,
    continuum_method='nanmedian',
    remove_invalid_spaxels=True,
    clean_invalid=True
):
    """
    Process a single IFS FITS file through the complete pipeline.
    
    Parameters
    ----------
    input_file : str or Path
        Path to input FITS file
    pixelscale_arcsec : float
        Pixel scale in arcseconds per pixel
    redshift : float
        Redshift for wavelength correction (z)
    instrument : str
        Instrument type ('sami', 'generic')
    continuum_file : str or Path, optional
        Path to continuum FITS file
    continuum_extension : str, optional
        Extension name for continuum data
    continuum_method : str
        Method for continuum subtraction ('nanmedian' or 'file')
    remove_invalid_spaxels : bool
        Whether to crop NaN edges
    clean_invalid : bool
        Whether to replace NaN/inf values
    
    Returns
    -------
    dict
        Processed data result dictionary
    """
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"Instrument: {instrument}")
    print(f"{'='*60}")
    
    # step 1: read IFS data
    print("Step 1: Reading IFS data...")
    if instrument.lower() == 'sami':
        data, weights, variance, header, wavelengths = read_sami_fits(input_file)
    elif instrument.lower() == 'generic':
        data, weights, variance, header, wavelengths = read_generic_ifs_fits(input_file)
    else:
        raise ValueError(f"Unknown instrument: {instrument}")
    
    print(f"  - Data cube shape: {data.shape}")
    print(f"  - Weights available: {'Yes' if weights is not None else 'No'}")
    print(f"  - Variance available: {'Yes' if variance is not None else 'No'}")
    print(f"  - Original wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} Å")
    
    # step 2: apply redshift correction
    if redshift != 0.0:
        print("\nStep 2: Applying redshift correction...")
        rest_wavelengths = redshift_correct_wavelengths(wavelengths, redshift)
    else:
        print("\nStep 2: No redshift correction applied (z=0)")
        rest_wavelengths = wavelengths
    
    # step 3: subtract continuum
    print("\nStep 3: Subtracting continuum...")
    if continuum_method == 'file' and continuum_file and continuum_extension:
        continuum = read_continuum_fits(continuum_file, continuum_extension)
        continuum_subtracted = subtract_continuum_from_file(data, continuum, weights)
        print(f"  - Using continuum from file: {continuum_file}")
    else:
        continuum_subtracted = subtract_continuum_nanmedian(data, weights)
        print(f"  - Using nanmedian across wavelength axis")
    
    # step 4: clean data
    if remove_invalid_spaxels:
        print("\nStep 4: Cleaning data (cropping NaN edges)...")
        cleaned_data, cleaned_weights, cleaned_variance, valid_mask, coord_info = clean_nan_edges(
            continuum_subtracted, pixelscale_arcsec, weights, variance)
        total_spaxels = data.shape[1] * data.shape[2]
        valid_spaxels = np.sum(valid_mask)
        print(f"  - Valid spaxels: {valid_spaxels}/{total_spaxels} ({100*valid_spaxels/total_spaxels:.1f}%)")
    else:
        print("\nStep 4: Skipping NaN edge cropping")
        cleaned_data = continuum_subtracted
        cleaned_weights = weights
        cleaned_variance = variance
        height, width = data.shape[-2], data.shape[-1]
        coord_info = {
            'x_min': -(width * pixelscale_arcsec) / 2.0,
            'x_max': (width * pixelscale_arcsec) / 2.0,
            'y_min': -(height * pixelscale_arcsec) / 2.0,
            'y_max': (height * pixelscale_arcsec) / 2.0,
            'spatial_sampling': pixelscale_arcsec
        }
    
    # step 5: clean invalid values
    if clean_invalid:
        print("\nStep 5: Cleaning invalid values...")
        cleaned_data, cleaned_weights, cleaned_variance = clean_invalid_values(
            cleaned_data, cleaned_weights, cleaned_variance)
    else:
        print("\nStep 5: Skipping invalid value cleaning")
    
    return {
        'data': cleaned_data,
        'weights': cleaned_weights,
        'variance': cleaned_variance,
        'wavelengths': rest_wavelengths,
        'coord_info': coord_info,
        'header': header,
        'instrument': instrument
    }


def process_windowed_ifs_data(
    input_files,
    output_dir,
    emission_lines=None,
    window_size=15.0,
    min_gap=5.0,
    resolution=None,
    pixelscale_arcsec=0.5,
    redshift=0.0,
    instrument='sami',
    continuum_files=None,
    continuum_extensions=None,
    continuum_method='nanmedian',
    remove_invalid_spaxels=True,
    clean_invalid=True,
    dry_run=False,
    create_plots=True
):
    """
    Complete workflow for processing IFS data into windowed format for Blobby3D.
    
    Parameters
    ----------
    input_files : str, Path, or list
        Input FITS file(s). For multi-arm data, provide list of files.
    output_dir : str or Path
        Output directory for processed data
    emission_lines : dict, optional
        Dictionary of emission lines {name: wavelength_angstrom}.
        If None, uses common optical lines.
    window_size : float
        Half-width of windows in Angstroms
    min_gap : float
        Minimum gap before combining windows (Angstroms)
    resolution : float or dict, optional
        Spectral resolution R = λ/Δλ
    pixelscale_arcsec : float
        Pixel scale in arcseconds per pixel
    redshift : float
        Redshift for wavelength correction
    instrument : str or list
        Instrument type(s)
    continuum_files : str, Path, or list, optional
        Continuum file(s) corresponding to input_files
    continuum_extensions : str or list, optional
        Extension name(s) for continuum data
    continuum_method : str
        Continuum subtraction method ('nanmedian' or 'file')
    remove_invalid_spaxels : bool
        Whether to crop NaN edges
    clean_invalid : bool
        Whether to replace NaN/inf values
    dry_run : bool
        If True, only create plots without writing data files
    create_plots : bool
        Whether to create diagnostic plots
    
    Returns
    -------
    dict
        Processing results and file paths
    """
    print("IFS Data Processing Workflow for Blobby3D")
    print("=" * 60)
    
    # handle single file vs multiple files
    if isinstance(input_files, (str, Path)):
        input_files = [input_files]
    if isinstance(instrument, str):
        instrument = [instrument] * len(input_files)
    if continuum_files and isinstance(continuum_files, (str, Path)):
        continuum_files = [continuum_files]
    if continuum_extensions and isinstance(continuum_extensions, str):
        continuum_extensions = [continuum_extensions] * len(input_files)
    
    # use default emission lines if not provided
    if emission_lines is None:
        emission_lines = {
            '[OII]3726': 3726.03,
            '[OII]3729': 3728.82,
            'H-beta': 4861.3,
            '[OIII]5007': 5006.8,
            'H-alpha': 6562.81,
            '[NII]6548': 6548.1,
            '[NII]6583': 6583.1,
            '[SII]6717': 6716.4,
            '[SII]6731': 6730.8
        }
    
    print(f"Input files: {len(input_files)}")
    for i, f in enumerate(input_files):
        print(f"  {i+1}: {f} ({instrument[i]})")
    print(f"Output directory: {output_dir}")
    print(f"Window size: ±{window_size} Å")
    print(f"Emission lines: {len(emission_lines)}")
    print(f"Redshift: {redshift}")
    print(f"Dry run: {dry_run}")
    print()
    
    # create output directory
    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # process individual files
        individual_results = []
        for i, input_file in enumerate(input_files):
            cont_file = continuum_files[i] if continuum_files else None
            cont_ext = continuum_extensions[i] if continuum_extensions else None
            
            result = process_single_ifs_file(
                input_file=input_file,
                pixelscale_arcsec=pixelscale_arcsec,
                redshift=redshift,
                instrument=instrument[i],
                continuum_file=cont_file,
                continuum_extension=cont_ext,
                continuum_method=continuum_method,
                remove_invalid_spaxels=remove_invalid_spaxels,
                clean_invalid=clean_invalid
            )
            individual_results.append(result)
        
        # combine multiple arms if needed
        if len(individual_results) > 1:
            print(f"\nCombining {len(individual_results)} datasets...")
            combined_result = combine_multi_arm_data(*individual_results)
        else:
            combined_result = individual_results[0]
            combined_result['individual_results'] = individual_results
        
        # define and process windows
        print(f"\nDefining wavelength windows...")
        windows = define_emission_line_windows(emission_lines, window_size)
        windows = filter_windows_by_coverage(windows, combined_result['wavelengths'])
        
        if not windows:
            raise ValueError("No emission lines found within the wavelength range!")
        
        windows = combine_overlapping_windows(windows, min_gap)
        windows = extract_wavelength_indices(windows, combined_result['wavelengths'], resolution)
        
        # extract windowed data
        windowed_data, windowed_var = extract_windowed_data(
            combined_result['data'], windows, combined_result['variance'])
        
        # write data files (unless dry run)
        written_files = {}
        if not dry_run:
            print(f"\nWriting Blobby3D format data...")
            data_file, var_file, weights_file = write_blobby_data(
                windowed_data, combined_result['weights'], windowed_var, output_dir)
            metadata_file = write_blobby_metadata(
                combined_result['coord_info'], windows, output_dir,
                windowed_data.shape[0], windowed_data.shape[1])
            
            written_files = {
                'data_file': data_file,
                'var_file': var_file,
                'weights_file': weights_file,
                'metadata_file': metadata_file
            }
        
        # create plots
        plot_files = {}
        if create_plots:
            print(f"\nCreating diagnostic plots...")
            
            if len(individual_results) > 1:
                # multi-arm comparison plot
                fig1 = create_multi_arm_comparison_plot(
                    individual_results, combined_result, windowed_data, windows, output_dir)
                plot_files['multi_arm_plot'] = output_path / 'multi_arm_windowed_comparison.png'
            
            # processing summary plot
            fig2 = create_processing_summary_plot(
                combined_result, windows, output_dir)
            plot_files['summary_plot'] = output_path / 'processing_summary.png'
        
        # print summary
        print_processing_summary(individual_results, combined_result, windows, windowed_data)
        
        # return results
        results = {
            'individual_results': individual_results,
            'combined_result': combined_result,
            'windows': windows,
            'windowed_data': windowed_data,
            'windowed_variance': windowed_var,
            'written_files': written_files,
            'plot_files': plot_files,
            'output_dir': output_path
        }
        
        if not dry_run:
            print(f"\nProcessing complete! Data saved to: {output_dir}")
            print(f"Files created:")
            for name, path in written_files.items():
                print(f"  - {path.name}")
        else:
            print(f"\nDry run complete! No data files written.")
        
        if create_plots:
            print(f"Diagnostic plots:")
            for name, path in plot_files.items():
                print(f"  - {path.name}")
        
        return results
        
    except Exception as e:
        print(f"Error in processing workflow: {e}")
        raise


def process_existing_blobby_data(
    input_dir,
    output_dir,
    emission_lines=None,
    window_size=15.0,
    min_gap=5.0,
    resolution=None,
    dry_run=False,
    create_plots=True,
    convert_old_format=True
):
    """
    Process existing Blobby3D format data to extract wavelength windows.
    
    This function is for re-processing already converted Blobby3D data
    with different windowing parameters. Automatically handles both old
    and new metadata formats.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing existing Blobby3D data
    output_dir : str or Path
        Output directory for windowed data
    emission_lines : dict, optional
        Dictionary of emission lines
    window_size : float
        Half-width of windows in Angstroms
    min_gap : float
        Minimum gap before combining windows
    resolution : float, optional
        Spectral resolution
    dry_run : bool
        If True, only create plots without writing files
    create_plots : bool
        Whether to create diagnostic plots
    convert_old_format : bool
        If True, automatically convert old format metadata to new format
    
    Returns
    -------
    dict
        Processing results
    """
    from ..io.blobby_io import load_blobby_metadata, load_blobby_data_cube, convert_old_to_new_metadata
    
    print("Blobby3D Data Re-windowing Workflow")
    print("=" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Window size: ±{window_size} Å")
    print(f"Dry run: {dry_run}")
    print()
    
    # file paths
    input_path = Path(input_dir)
    data_file = input_path / 'data.txt'
    var_file = input_path / 'var.txt'
    metadata_file = input_path / 'metadata.txt'
    
    # check required files exist
    required_files = [data_file, metadata_file]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # load existing data and handle format conversion
    metadata = load_blobby_metadata(metadata_file)
    
    # if old format and conversion enabled, convert to new format
    if metadata.get('format') == 'old' and convert_old_format and not dry_run:
        print("\nDetected old format metadata. Converting to new format...")
        new_metadata_file = convert_old_to_new_metadata(
            metadata_file, 
            Path(output_dir) / 'metadata_converted.txt',
            resolution=resolution
        )
        print(f"Converted metadata saved as: {new_metadata_file.name}")
        if resolution is not None:
            print(f"Added resolution R = {resolution:.0f} to converted metadata")
    
    original_data = load_blobby_data_cube(data_file, metadata)
    
    original_var = None
    if var_file.exists():
        original_var = load_blobby_data_cube(var_file, metadata)
        print(f"Loaded variance data from: {var_file}")
    
    # use default emission lines if not provided
    if emission_lines is None:
        emission_lines = {
            '[OII]3726': 3726.03,
            '[OII]3729': 3728.82,
            'H-beta': 4861.3,
            '[OIII]5007': 5006.8,
            'H-alpha': 6562.81,
            '[NII]6548': 6548.1,
            '[NII]6583': 6583.1,
            '[SII]6717': 6716.4,
            '[SII]6731': 6730.8
        }
    
    # define windows
    print(f"\nDefining wavelength windows...")
    windows = define_emission_line_windows(emission_lines, window_size)
    windows = filter_windows_by_coverage(windows, metadata)
    
    if not windows:
        raise ValueError("No emission lines found within the wavelength range!")
    
    windows = combine_overlapping_windows(windows, min_gap)
    windows = extract_wavelength_indices(windows, metadata['wavelengths'], resolution)
    
    # extract windowed data (need to transpose from Blobby format to processing format)
    data_for_windowing = original_data.transpose(2, 0, 1)  # (ni, nj, nr) -> (nr, ni, nj)
    var_for_windowing = original_var.transpose(2, 0, 1) if original_var is not None else None
    
    windowed_data, windowed_var = extract_windowed_data(
        data_for_windowing, windows, var_for_windowing)
    
    # create output directory and write files
    output_path = Path(output_dir)
    written_files = {}
    
    if not dry_run:
        print(f"\nWriting windowed Blobby3D data...")
        data_file_out, var_file_out, weights_file_out = write_blobby_data(
            windowed_data, None, windowed_var, output_dir)
        
        # create coordinate info from metadata
        coord_info = {
            'x_min': metadata['x_min'],
            'x_max': metadata['x_max'],
            'y_min': metadata['y_min'],
            'y_max': metadata['y_max'],
            'spatial_sampling': (metadata['x_max'] - metadata['x_min']) / metadata['ni']
        }
        
        metadata_file_out = write_blobby_metadata(
            coord_info, windows, output_dir,
            windowed_data.shape[0], windowed_data.shape[1])
        
        written_files = {
            'data_file': data_file_out,
            'var_file': var_file_out,
            'metadata_file': metadata_file_out
        }
        
        # add converted metadata file if created
        if metadata.get('format') == 'old' and convert_old_format:
            written_files['converted_metadata_file'] = output_path / 'metadata_converted.txt'
    
    # create plots
    plot_files = {}
    if create_plots:
        print(f"\nCreating comparison plots...")
        fig = create_comparison_plot(
            original_data, windowed_data, metadata, windows, output_dir)
        plot_files['comparison_plot'] = output_path / 'windowed_data_comparison.png'
    
    # print summary
    print(f"\n{'='*60}")
    print("RE-WINDOWING SUMMARY")
    print(f"{'='*60}")
    
    orig_shape = original_data.shape
    wind_shape = windowed_data.shape
    
    print(f"Original data shape: {orig_shape}")
    print(f"Windowed data shape: {wind_shape}")
    print(f"Data reduction: {orig_shape[2]} → {wind_shape[2]} bins ({100*wind_shape[2]/orig_shape[2]:.1f}%)")
    
    if metadata.get('format') == 'old':
        print(f"Metadata format: Old format (converted to new format)")
    else:
        print(f"Metadata format: New format")
    
    print(f"\nWindow details:")
    for i, window in enumerate(windows):
        actual_min = window.get('actual_r_min', window['r_min'])
        actual_max = window.get('actual_r_max', window['r_max'])
        n_bins = window.get('n_bins', 0)
        print(f"  Window {i+1}: {window['name']}")
        print(f"    {actual_min:.1f} - {actual_max:.1f} Å ({n_bins} bins)")
    
    results = {
        'metadata': metadata,
        'original_data': original_data,
        'original_variance': original_var,
        'windows': windows,
        'windowed_data': windowed_data,
        'windowed_variance': windowed_var,
        'written_files': written_files,
        'plot_files': plot_files,
        'output_dir': output_path,
        'format_converted': metadata.get('format') == 'old' and convert_old_format
    }
    
    if not dry_run:
        print(f"\nRe-windowing complete! Data saved to: {output_dir}")
        if results['format_converted']:
            print("Old format metadata was automatically converted to new format.")
    else:
        print(f"\nDry run complete! No files written.")
    
    return results

def print_processing_summary(individual_results, combined_result, windows, windowed_data):
    """
    Print a comprehensive summary of the processing results.
    
    Parameters
    ----------
    individual_results : list
        List of individual dataset results
    combined_result : dict
        Combined dataset result
    windows : list
        List of window dictionaries
    windowed_data : np.ndarray
        Final windowed data
    """
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    # dataset information
    print(f"Number of input datasets: {len(individual_results)}")
    for i, result in enumerate(individual_results):
        shape = result['data'].shape
        waves = result['wavelengths']
        print(f"  Dataset {i+1} ({result.get('instrument', 'unknown')}):")
        print(f"    Shape: {shape}")
        print(f"    Wavelength range: {waves[0]:.1f} - {waves[-1]:.1f} Å")
    
    # combined data information
    combined_shape = combined_result['data'].shape
    combined_waves = combined_result['wavelengths']
    windowed_shape = windowed_data.shape
    
    print(f"\nCombined data:")
    print(f"  Shape: {combined_shape}")
    print(f"  Wavelength range: {combined_waves[0]:.1f} - {combined_waves[-1]:.1f} Å")
    print(f"  Total bins: {combined_shape[0]}")
    
    print(f"\nWindowed data:")
    print(f"  Shape: {windowed_shape} (y, x, wavelength)")
    print(f"  Total windowed bins: {windowed_shape[2]}")
    print(f"  Data reduction: {combined_shape[0]} → {windowed_shape[2]} bins ({100*windowed_shape[2]/combined_shape[0]:.1f}%)")
    
    # spatial coverage
    coord_info = combined_result['coord_info']
    print(f"\nSpatial coverage:")
    print(f"  {windowed_shape[0]} × {windowed_shape[1]} spaxels")
    print(f"  X: {coord_info['x_min']:.1f}″ to {coord_info['x_max']:.1f}″")
    print(f"  Y: {coord_info['y_min']:.1f}″ to {coord_info['y_max']:.1f}″")
    print(f"  Pixel scale: {coord_info['spatial_sampling']:.2f} arcsec/pixel")
    
    # window information
    print(f"\nWavelength windows: {len(windows)}")
    total_coverage = sum(w.get('actual_width', w['width']) for w in windows)
    total_range = combined_waves[-1] - combined_waves[0]
    print(f"  Total coverage: {total_coverage:.1f} Å ({100*total_coverage/total_range:.1f}% of range)")
    
    for i, window in enumerate(windows):
        actual_min = window.get('actual_r_min', window['r_min'])
        actual_max = window.get('actual_r_max', window['r_max'])
        n_bins = window.get('n_bins', 0)
        resolution = window.get('resolution', 'N/A')
        
        print(f"  Window {i+1}: {window['name']}")
        print(f"    Range: {actual_min:.1f} - {actual_max:.1f} Å ({n_bins} bins)")
        if resolution != 'N/A':
            print(f"    Resolution: R = {resolution:.0f}")
        if 'lines' in window:
            print(f"    Lines: {', '.join(window['lines'])}")
    
    print("="*80)


# convenience functions for common tasks
def quick_sami_processing(blue_file, red_file, output_dir, redshift=0.0, **kwargs):
    """
    Quick function for standard SAMI blue+red processing.
    
    Parameters
    ----------
    blue_file : str or Path
        SAMI blue FITS file
    red_file : str or Path  
        SAMI red FITS file
    output_dir : str or Path
        Output directory
    redshift : float
        Target redshift
    **kwargs
        Additional arguments passed to process_windowed_ifs_data
    
    Returns
    -------
    dict
        Processing results
    """
    return process_windowed_ifs_data(
        input_files=[blue_file, red_file],
        output_dir=output_dir,
        redshift=redshift,
        instrument=['sami', 'sami'],
        **kwargs
    )


def quick_single_ifs_processing(fits_file, output_dir, instrument='generic', **kwargs):
    """
    Quick function for single IFS file processing.
    
    Parameters
    ----------
    fits_file : str or Path
        IFS FITS file
    output_dir : str or Path
        Output directory
    instrument : str
        Instrument type ('sami', 'generic')
    **kwargs
        Additional arguments passed to process_windowed_ifs_data
    
    Returns
    -------
    dict
        Processing results
    """
    return process_windowed_ifs_data(
        input_files=fits_file,
        output_dir=output_dir,
        instrument=instrument,
        **kwargs
    )


def batch_process_ifs_data(file_list, output_base_dir, **kwargs):
    """
    Process multiple IFS datasets in batch.
    
    Parameters
    ----------
    file_list : list
        List of (input_files, output_subdir) tuples
    output_base_dir : str or Path
        Base output directory
    **kwargs
        Common arguments passed to process_windowed_ifs_data
    
    Returns
    -------
    list
        List of processing results for each dataset
    """
    results = []
    base_path = Path(output_base_dir)
    
    print(f"Batch processing {len(file_list)} datasets...")
    
    for i, (input_files, subdir) in enumerate(file_list):
        print(f"\n{'='*60}")
        print(f"Processing dataset {i+1}/{len(file_list)}: {subdir}")
        print(f"{'='*60}")
        
        output_dir = base_path / subdir
        
        try:
            result = process_windowed_ifs_data(
                input_files=input_files,
                output_dir=output_dir,
                **kwargs
            )
            results.append(result)
            print(f"SUCCESS: {subdir}")
            
        except Exception as e:
            print(f"FAILED: {subdir} - {e}")
            results.append(None)
    
    # print batch summary
    successful = sum(1 for r in results if r is not None)
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(file_list)}")
    print(f"Failed: {len(file_list) - successful}/{len(file_list)}")
    
    return results


def validate_processing_inputs(input_files, continuum_files=None, required_extensions=None):
    """
    Validate that all required input files exist before processing.
    
    Parameters
    ----------
    input_files : str, Path, or list
        Input FITS files
    continuum_files : str, Path, or list, optional
        Continuum FITS files
    required_extensions : list, optional
        List of required FITS extensions to check
    
    Raises
    ------
    FileNotFoundError
        If any required files are missing
    ValueError
        If FITS files are invalid or missing required extensions
    """
    from astropy.io import fits
    
    # convert to lists
    if isinstance(input_files, (str, Path)):
        input_files = [input_files]
    if continuum_files and isinstance(continuum_files, (str, Path)):
        continuum_files = [continuum_files]
    
    print("Validating input files...")
    
    # check input files exist
    for file_path in input_files:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        print(f"  ✓ {path.name}")
    
    # check continuum files if provided
    if continuum_files:
        for file_path in continuum_files:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Continuum file not found: {file_path}")
            print(f"  ✓ {path.name} (continuum)")
    
    # check FITS extensions if specified
    if required_extensions:
        for file_path in input_files:
            try:
                with fits.open(file_path) as hdul:
                    available_extensions = [hdu.header.get('EXTNAME', f'HDU{i}') for i, hdu in enumerate(hdul)]
                    
                for ext in required_extensions:
                    if ext not in available_extensions:
                        print(f"  ⚠ Warning: Extension '{ext}' not found in {Path(file_path).name}")
                        print(f"    Available: {available_extensions}")
            except Exception as e:
                raise ValueError(f"Cannot read FITS file {file_path}: {e}")
    
    print("Input validation complete.")


def safe_process_with_logging(processing_func, log_file=None, **kwargs):
    """
    Wrapper for processing functions with enhanced error handling and logging.
    
    Parameters
    ----------
    processing_func : callable
        Processing function to run
    log_file : str or Path, optional
        Path to log file for output
    **kwargs
        Arguments passed to processing_func
    
    Returns
    -------
    dict or None
        Processing results, or None if failed
    """
    import sys
    import traceback
    from datetime import datetime
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_message(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')
    
    log_message("Starting IFS processing...")
    
    try:
        # run the processing function
        result = processing_func(**kwargs)
        log_message("Processing completed successfully")
        return result
        
    except FileNotFoundError as e:
        log_message(f"ERROR: File not found - {e}")
        return None
        
    except ValueError as e:
        log_message(f"ERROR: Invalid input - {e}")
        return None
        
    except Exception as e:
        log_message(f"ERROR: Processing failed - {e}")
        log_message("Full traceback:")
        
        # log full traceback
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        for line in tb_lines:
            log_message(line.rstrip())
        
        return None


def get_processing_status(output_dir):
    """
    Check the status of processing in an output directory.
    
    Parameters
    ----------
    output_dir : str or Path
        Output directory to check
    
    Returns
    -------
    dict
        Status information about the processing
    """
    output_path = Path(output_dir)
    
    status = {
        'directory_exists': output_path.exists(),
        'data_file_exists': False,
        'metadata_file_exists': False,
        'var_file_exists': False,
        'weights_file_exists': False,
        'plots_exist': [],
        'processing_complete': False
    }
    
    if not output_path.exists():
        return status
    
    # check for required files
    data_file = output_path / 'data.txt'
    metadata_file = output_path / 'metadata.txt'
    var_file = output_path / 'var.txt'
    weights_file = output_path / 'weights.txt'
    
    status['data_file_exists'] = data_file.exists()
    status['metadata_file_exists'] = metadata_file.exists()
    status['var_file_exists'] = var_file.exists()
    status['weights_file_exists'] = weights_file.exists()
    
    # check for plots
    plot_patterns = [
        'multi_arm_windowed_comparison.png',
        'processing_summary.png',
        'windowed_data_comparison.png'
    ]
    
    for pattern in plot_patterns:
        plot_file = output_path / pattern
        if plot_file.exists():
            status['plots_exist'].append(pattern)
    
    # determine if processing is complete
    status['processing_complete'] = (
        status['data_file_exists'] and 
        status['metadata_file_exists']
    )
    
    return status


def cleanup_processing_outputs(output_dir, keep_plots=True):
    """
    Clean up processing outputs, optionally keeping plots.
    
    Parameters
    ----------
    output_dir : str or Path
        Output directory to clean
    keep_plots : bool
        Whether to keep diagnostic plots
    
    Returns
    -------
    list
        List of files that were removed
    """
    output_path = Path(output_dir)
    removed_files = []
    
    if not output_path.exists():
        print(f"Directory does not exist: {output_dir}")
        return removed_files
    
    # files to remove
    files_to_remove = ['data.txt', 'var.txt', 'weights.txt', 'metadata.txt']
    
    if not keep_plots:
        files_to_remove.extend([
            'multi_arm_windowed_comparison.png',
            'processing_summary.png', 
            'windowed_data_comparison.png'
        ])
    
    for filename in files_to_remove:
        file_path = output_path / filename
        if file_path.exists():
            file_path.unlink()
            removed_files.append(str(file_path))
            print(f"Removed: {filename}")
    
    print(f"Cleanup complete. Removed {len(removed_files)} files.")
    return removed_files