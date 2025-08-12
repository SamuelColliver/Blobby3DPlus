#!/usr/bin/env python3
"""
High-level processing workflow functions with FIXED LSF FWHM handling.
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
    """Process a single IFS FITS file through the complete pipeline."""
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
    instrument_resolutions=None,    # new: per-instrument resolutions
    instrument_ranges=None,         # new: per-instrument wavelength ranges
    resolution=None,                # single resolution (for backward compatibility)
    lsf_fwhm=None,                 # single LSF FWHM (for backward compatibility)
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
    """Complete workflow for processing IFS data with per-window LSF FWHM calculation."""
    print("IFS Data Processing Workflow with Per-Window LSF FWHM")
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
    
    # print resolution setup
    if instrument_resolutions and instrument_ranges:
        print(f"Multi-instrument setup:")
        for instr in instrument_resolutions:
            r_min, r_max = instrument_ranges[instr]
            print(f"  {instr}: R = {instrument_resolutions[instr]}, λ = {r_min}-{r_max} Å")
    elif resolution is not None:
        print(f"Single resolution: R = {resolution}")
    elif lsf_fwhm is not None:
        print(f"Fixed LSF FWHM: {lsf_fwhm:.3f} Å")
    
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
        
        # prepare multi-instrument setup for LSF FWHM calculation
        multi_instrument_setup = None
        if instrument_resolutions and instrument_ranges:
            multi_instrument_setup = {
                'resolutions': instrument_resolutions,
                'wavelength_ranges': instrument_ranges
            }
        
        # extract wavelength indices with per-window LSF FWHM calculation
        windows = extract_wavelength_indices(
            windows,
            combined_result['wavelengths'],
            multi_instrument_setup=multi_instrument_setup,
            resolution=resolution,
            lsf_fwhm=lsf_fwhm,
            redshift=redshift
        )
        
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
        
        # print summary with per-window LSF FWHM info
        print_processing_summary_with_per_window_lsf(individual_results, combined_result, windows, windowed_data)
        
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
    lsf_fwhm=None,          # for old format conversion
    resolution=None,        # for old format conversion  
    dry_run=False,
    create_plots=True,
    convert_old_format=True
):

    from ..io.blobby_io import (
        load_blobby_metadata, load_blobby_data_cube, 
        convert_old_to_new_metadata, write_blobby_data, write_blobby_metadata
    )
    from ..processing.windowing import (
        define_emission_line_windows, filter_windows_by_coverage,
        combine_overlapping_windows, extract_windowed_data
    )
    
    print("Blobby3D Data Re-windowing")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Window size: ±{window_size} Å")
    print(f"Minimum gap: {min_gap} Å")
    
    if lsf_fwhm is not None:
        print(f"LSF FWHM (for old format): {lsf_fwhm} Å")
    elif resolution is not None:
        print(f"Resolution (for old format): R = {resolution}")
    
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
    
    # FIXED: Handle old format conversion and LSF FWHM assignment
    conversion_lsf_fwhm = None
    format_converted = False
    
    if metadata.get('format') == 'old' and convert_old_format:
        print("\nDetected old format metadata. Converting...")
        
        # calculate LSF FWHM for conversion
        if lsf_fwhm is not None:
            conversion_lsf_fwhm = lsf_fwhm
            print(f"Using provided LSF FWHM: {lsf_fwhm} Å")
        elif resolution is not None:
            # calculate LSF FWHM from central wavelength and resolution
            central_wavelength = (metadata['r_min'] + metadata['r_max']) / 2.0
            conversion_lsf_fwhm = central_wavelength / resolution
            print(f"Calculated LSF FWHM: {conversion_lsf_fwhm:.3f} Å (λ={central_wavelength:.1f} Å, R={resolution})")
        else:
            print("Warning: No LSF FWHM or resolution provided for old format conversion")
        
        # convert metadata to new format (but don't write yet)
        if not dry_run:
            new_metadata_file = Path(output_dir) / 'metadata_converted.txt'
            new_metadata_file.parent.mkdir(parents=True, exist_ok=True)
            convert_old_to_new_metadata(
                metadata_file, 
                new_metadata_file,
                lsf_fwhm=lsf_fwhm,
                resolution=resolution
            )
            print(f"Converted metadata saved as: {new_metadata_file.name}")
        
        format_converted = True
    
    # load data
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
    
    # FIXED: Properly assign wavelength indices AND LSF FWHM
    print(f"\nMapping windows and assigning LSF FWHM...")
    windows = extract_wavelength_indices_for_existing_data(
        windows, 
        metadata['wavelengths'],
        metadata,
        conversion_lsf_fwhm
    )
    
    # extract windowed data
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
        
        # FIXED: Write metadata with LSF FWHM values
        metadata_file_out = write_blobby_metadata(
            coord_info, windows, output_dir,
            windowed_data.shape[0], windowed_data.shape[1])
        
        written_files = {
            'data_file': data_file_out,
            'var_file': var_file_out,
            'metadata_file': metadata_file_out
        }
        
        # add converted metadata file if created
        if format_converted:
            written_files['converted_metadata_file'] = output_path / 'metadata_converted.txt'
    
    # create plots
    plot_files = {}
    if create_plots:
        print(f"\nCreating comparison plots...")
        fig = create_comparison_plot(
            original_data, windowed_data, metadata, windows, output_dir)
        plot_files['comparison_plot'] = output_path / 'windowed_data_comparison.png'
    
    # print summary with per-window LSF FWHM
    print_rewindowing_summary_with_per_window_lsf(metadata, windows, original_data, windowed_data)
    
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
        'format_converted': format_converted,
        'conversion_lsf_fwhm': conversion_lsf_fwhm
    }
    
    if not dry_run:
        print(f"\nRe-windowing complete! Data saved to: {output_dir}")
        if results['format_converted']:
            print("Old format metadata was automatically converted to new format.")
    else:
        print(f"\nDry run complete! No files written.")
    
    return results


def extract_wavelength_indices_for_existing_data(windows, wavelengths, metadata, conversion_lsf_fwhm):
    print(f"Mapping windows and assigning LSF FWHM...")
    
    # extract wavelength indices first
    for window in windows:
        # find closest indices in the wavelength array
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
    
    # FIXED: Assign LSF FWHM based on metadata format and available information
    if metadata.get('format') == 'new' and 'wavelength_ranges' in metadata:
        # new format: propagate from existing per-window LSF FWHM
        assign_lsf_from_existing_metadata(windows, metadata)
    elif conversion_lsf_fwhm is not None:
        # old format or explicit conversion: use conversion LSF FWHM for all windows
        assign_lsf_from_conversion(windows, conversion_lsf_fwhm)
    else:
        # no LSF FWHM information available
        print("Warning: No LSF FWHM information available")
        for window in windows:
            window['lsf_fwhm'] = None
    
    return windows


def assign_lsf_from_existing_metadata(windows, metadata):
    """Assign LSF FWHM to new windows based on existing metadata with per-window values."""
    print("Propagating LSF FWHM from existing new-format metadata:")
    
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
        print("  Warning: No LSF FWHM values found in existing metadata")
        for window in windows:
            window['lsf_fwhm'] = None
        return
    
    print(f"  Found {len(existing_lsf_data)} existing LSF FWHM values:")
    for data in existing_lsf_data:
        print(f"    λ={data['wavelength']:.1f} Å: LSF FWHM={data['lsf_fwhm']:.3f} Å")
    
    # assign LSF FWHM to new windows based on closest wavelength
    for window in windows:
        central_wave = (window['actual_r_min'] + window['actual_r_max']) / 2.0
        
        # find the closest existing LSF FWHM value by wavelength
        distances = [abs(central_wave - data['wavelength']) for data in existing_lsf_data]
        closest_idx = np.argmin(distances)
        closest_data = existing_lsf_data[closest_idx]
        
        window['lsf_fwhm'] = closest_data['lsf_fwhm']
        
        print(f"  {window['name']} (λ={central_wave:.1f} Å): LSF FWHM = {window['lsf_fwhm']:.3f} Å")
        print(f"    (from existing λ={closest_data['wavelength']:.1f} Å)")


def assign_lsf_from_conversion(windows, conversion_lsf_fwhm):
    """Assign the same conversion LSF FWHM to all new windows."""
    print(f"Assigning conversion LSF FWHM to all windows:")
    print(f"  LSF FWHM = {conversion_lsf_fwhm:.3f} Å (from old format conversion)")
    
    for window in windows:
        window['lsf_fwhm'] = conversion_lsf_fwhm
        central_wave = (window['actual_r_min'] + window['actual_r_max']) / 2.0
        print(f"  {window['name']} (λ={central_wave:.1f} Å): LSF FWHM = {window['lsf_fwhm']:.3f} Å")


def print_processing_summary_with_per_window_lsf(individual_results, combined_result, windows, windowed_data):
    """Print summary with per-window LSF FWHM information."""
    print("\n" + "="*80)
    print("PROCESSING SUMMARY (Per-Window LSF FWHM)")
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
    
    # per-window LSF FWHM information
    print(f"\nWavelength windows with per-window LSF FWHM: {len(windows)}")
    total_coverage = sum(w.get('actual_width', w['width']) for w in windows)
    total_range = combined_waves[-1] - combined_waves[0]
    print(f"  Total coverage: {total_coverage:.1f} Å ({100*total_coverage/total_range:.1f}% of range)")
    
    for i, window in enumerate(windows):
        actual_min = window.get('actual_r_min', window['r_min'])
        actual_max = window.get('actual_r_max', window['r_max'])
        n_bins = window.get('n_bins', 0)
        lsf_fwhm = window.get('lsf_fwhm', None)
        instrument = window.get('instrument', 'unknown')
        resolution = window.get('resolution', None)
        central_obs = window.get('central_wavelength_observed', None)
        
        print(f"  Window {i+1}: {window['name']}")
        print(f"    Rest-frame range: {actual_min:.1f} - {actual_max:.1f} Å ({n_bins} bins)")
        if central_obs is not None:
            print(f"    Central λ (observed): {central_obs:.1f} Å")
        if instrument != 'unknown' and resolution is not None:
            print(f"    Instrument: {instrument} (R = {resolution})")
        if lsf_fwhm is not None:
            print(f"    LSF FWHM: {lsf_fwhm:.3f} Å")
        else:
            print(f"    LSF FWHM: Not available")
        if 'lines' in window:
            print(f"    Lines: {', '.join(window['lines'])}")
    
    print("="*80)


def print_rewindowing_summary_with_per_window_lsf(metadata, windows, original_data, windowed_data):
    """Print re-windowing summary with per-window LSF FWHM information."""
    print(f"\n{'='*60}")
    print("RE-WINDOWING SUMMARY (Per-Window LSF FWHM)")
    print(f"{'='*60}")
    
    orig_shape = original_data.shape
    wind_shape = windowed_data.shape
    
    print(f"Original data shape: {orig_shape}")
    print(f"Windowed data shape: {wind_shape}")
    print(f"Data reduction: {orig_shape[2]} → {wind_shape[2]} bins ({100*wind_shape[2]/orig_shape[2]:.1f}%)")
    
    if metadata.get('format') == 'old':
        print(f"Metadata format: Old format (converted to new format)")
    else:
        print(f"Metadata format: New format with per-window LSF FWHM")
    
    print(f"\nWindow details with LSF FWHM:")
    for i, window in enumerate(windows):
        actual_min = window.get('actual_r_min', window['r_min'])
        actual_max = window.get('actual_r_max', window['r_max'])
        n_bins = window.get('n_bins', 0)
        lsf_fwhm = window.get('lsf_fwhm', None)
        
        print(f"  Window {i+1}: {window['name']}")
        print(f"    Range: {actual_min:.1f} - {actual_max:.1f} Å ({n_bins} bins)")
        if lsf_fwhm is not None:
            print(f"    LSF FWHM: {lsf_fwhm:.3f} Å")
        else:
            print(f"    LSF FWHM: Not available")
    
    print("="*60)


# convenience functions for common tasks
def quick_sami_processing(blue_file, red_file, output_dir, redshift=0.0, **kwargs):
    """Quick function for standard SAMI blue+red processing."""
    return process_windowed_ifs_data(
        input_files=[blue_file, red_file],
        output_dir=output_dir,
        redshift=redshift,
        instrument=['sami', 'sami'],
        **kwargs
    )


def quick_single_ifs_processing(fits_file, output_dir, instrument='generic', **kwargs):
    """Quick function for single IFS file processing."""
    return process_windowed_ifs_data(
        input_files=fits_file,
        output_dir=output_dir,
        instrument=instrument,
        **kwargs
    )