#!/usr/bin/env python3
"""
Example script for processing combined SAMI blue+red IFS data with per-window LSF FWHM.
"""

import sys
from pathlib import Path

# add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import process_windowed_ifs_data


def main():
    """Process SAMI blue and red IFS data into windowed format for Blobby3D."""
    
    # +++++++++++++ Configuration - Edit This Section +++++++++++++++++
    
    

    # input files
    input_files = [
        '/Users/scol0322/Documents/raw-data/485885/485885_blue_8_Y13SAR1_P007_15T010_2013_03_05-2013_03_17.fits',
        '/Users/scol0322/Documents/raw-data/485885/485885_red_8_Y13SAR1_P007_15T010_2013_03_05-2013_03_17.fits'
    ]
    
    # continuum files (optional - set to None to use nanmedian)
    continuum_files = [
        '/Users/scol0322/Documents/raw-data/485885/485885_8_Y13SAR1_P007_15T010_2013_03_05-2013_03_17_1_comp.fits',
        '/Users/scol0322/Documents/raw-data/485885/485885_8_Y13SAR1_P007_15T010_2013_03_05-2013_03_17_1_comp.fits'
    ]
    
    # continuum extensions (if using continuum files)
    continuum_extensions = ['B_CONTINUUM', 'R_CONTINUUM']
    
    # output directory
    output_dir = '/Users/scol0322/Documents/output/485885-combined'
    
    # processing parameters
    pixelscale_arcsec = 0.5        # SAMI spatial sampling
    redshift = 0.055            # target redshift
    
    # windowing parameters
    window_size = 15.0             # half-width in Angstroms (±15 Å = 30 Å total)
    min_gap = 5.0                  # combine windows closer than 5 Å
    

    # Each window will get its own LSF FWHM based on which instrument covers it
    instrument_resolutions = {
        'blue': 1812,              # R for SAMI blue arm
        'red': 4326                # R for SAMI red arm  
    }
    
    # wavelength ranges covered by each instrument (observed frame)
    instrument_ranges = {
        'blue': (3700, 5500),      # blue arm coverage in observed wavelengths
        'red': (5500, 7500)        # red arm coverage in observed wavelengths
    }
    
    # processing options
    instruments = [
        'sami', 
        'sami'
    ] # one per input file

    continuum_method = 'file'      # 'file' or 'nanmedian'
    remove_invalid_spaxels = True  # crop NaN edges
    clean_invalid = True           # replace NaN/inf values with zero
    
    # output options
    dry_run = False                # set True for plots only, no data files
    create_plots = True            # create diagnostic plots
    
    # emission lines to target (rest wavelengths in Angstroms)
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
    
    # +++++++++ End of Configuration +++++++++++++
    
    # validate input files
    for file in input_files:
        if not Path(file).exists():
            print(f"Error: Input file not found: {file}")
            sys.exit(1)
    
    if continuum_method == 'file' and continuum_files:
        for file in continuum_files:
            if not Path(file).exists():
                print(f"Error: Continuum file not found: {file}")
                sys.exit(1)
    
    print("SAMI Combined Processing with Per-Window LSF FWHM")
    print("=" * 60)
    print(f"Blue file: {Path(input_files[0]).name}")
    print(f"Red file: {Path(input_files[1]).name}")
    print(f"Output: {output_dir}")
    print(f"Redshift: {redshift}")
    print()
    
    print("Instrument setup:")
    print(f"  Blue arm: R = {instrument_resolutions['blue']}, λ = {instrument_ranges['blue'][0]}-{instrument_ranges['blue'][1]} Å")
    print(f"  Red arm: R = {instrument_resolutions['red']}, λ = {instrument_ranges['red'][0]}-{instrument_ranges['red'][1]} Å")
    print()
    
    # run processing pipeline with NEW per-window LSF FWHM approach
    try:
        results = process_windowed_ifs_data(
            input_files=input_files,
            output_dir=output_dir,
            emission_lines=emission_lines,
            window_size=window_size,
            min_gap=min_gap,
            
            # NEW: per-instrument setup for LSF FWHM calculation
            instrument_resolutions=instrument_resolutions,
            instrument_ranges=instrument_ranges,
            
            pixelscale_arcsec=pixelscale_arcsec,
            redshift=redshift,
            instrument=instruments,
            continuum_files=continuum_files if continuum_method == 'file' else None,
            continuum_extensions=continuum_extensions if continuum_method == 'file' else None,
            continuum_method=continuum_method,
            remove_invalid_spaxels=remove_invalid_spaxels,
            clean_invalid=clean_invalid,
            dry_run=dry_run,
            create_plots=create_plots
        )
        
        print("\n" + "="*60)
        print("SUCCESS: SAMI Combined Processing Complete")
        print("="*60)
        
        # show the actual calculated LSF FWHM for each window
        print("\nActual per-window LSF FWHM calculated:")
        for window in results['windows']:
            if 'lsf_fwhm' in window and window['lsf_fwhm'] is not None:
                central_obs = window.get('central_wavelength_observed', 0)
                instrument = window.get('instrument', 'unknown')
                resolution = window.get('resolution', 0)
                print(f"  {window['name']}: {central_obs:.0f} Å → {instrument} (R={resolution}) → LSF FWHM = {window['lsf_fwhm']:.3f} Å")
        
        if not dry_run:
            print("\nOutput files ready for Blobby3D:")
            for name, path in results['written_files'].items():
                if path:  # some might be None
                    print(f"  - {path}")
            
            print(f"\nMetadata file contains per-window LSF FWHM values.")
            print(f"Each wave_range line has its own LSF FWHM parameter.")
        
        if create_plots:
            print("\nDiagnostic plots created:")
            for name, path in results['plot_files'].items():
                print(f"  - {path}")
        
        print(f"\nKey improvement: Each window now has its own LSF FWHM!")
        print(f"Blue arm windows: ~2.1-2.9 Å LSF FWHM (lower resolution)")
        print(f"Red arm windows: ~1.6 Å LSF FWHM (higher resolution)")
        
    except Exception as e:
        print(f"\nERROR: Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
