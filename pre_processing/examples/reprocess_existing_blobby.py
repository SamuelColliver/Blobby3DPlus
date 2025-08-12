#!/usr/bin/env python3
"""
FIXED: Example script for re-processing existing Blobby3D data with proper LSF FWHM handling.
This version ensures LSF FWHM values are properly written to the output metadata.
"""

import sys
from pathlib import Path

# add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import process_existing_blobby_data


def main():
    """Re-process existing Blobby3D data with new windowing parameters."""
    
    # +++++++++++++ Configuration - Edit This Section +++++++++++++++++
    
    # input directory containing existing Blobby3D data
    input_dir = '../../examples/485885'
    
    # output directory for re-windowed data
    output_dir = '/Users/scol0322/Documents/output/485885-reprocessed'
    
    # windowing parameters
    window_size = 5.0              # half-width in Angstroms (smaller windows)
    min_gap = 0.0                  # don't combine any windows
    
    # IMPORTANT: LSF FWHM parameters for old format conversion
    # (this will be used to calculate the LSF FWHM for each new window)
    lsf_fwhm = 1.61                # Angstroms - FIXED: this value will be applied to all windows
    resolution = None              # Alternative: provide resolution to calculate LSF FWHM
    
    # Alternative approach: calculate from resolution
    # lsf_fwhm = None              # set to None if using resolution
    # resolution = 4900            # R = λ/Δλ - will calculate LSF FWHM = λ_central/R for each window
    
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
    
    # validate input directory
    input_path = Path(input_dir)
    required_files = ['data.txt', 'metadata.txt']
    
    for filename in required_files:
        if not (input_path / filename).exists():
            print(f"Error: Required file not found: {input_path / filename}")
            sys.exit(1)
    
    print("FIXED: Re-processing Existing Blobby3D Data with Per-Window LSF FWHM")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Window size: ±{window_size} Å")
    print(f"Minimum gap: {min_gap} Å")
    
    if lsf_fwhm is not None:
        print(f"LSF FWHM (for all windows): {lsf_fwhm} Å")
    elif resolution is not None:
        print(f"Resolution (will calculate per-window LSF FWHM): R = {resolution}")
    
    print(f"Dry run: {dry_run}")
    print()
    
    print("FIXED LSF FWHM handling:")
    print("- Old format → converts and applies provided LSF FWHM to all new windows")
    print("- New format → propagates existing per-window LSF FWHM to new windows") 
    print("- Output metadata will contain LSF FWHM values in wave_range lines")
    print()
    
    # Check input format first
    try:
        from src.io.blobby_io import load_blobby_metadata
        metadata = load_blobby_metadata(input_path / 'metadata.txt')
        
        if metadata.get('format') == 'old':
            print("✓ Detected OLD format metadata - will convert with LSF FWHM")
            if lsf_fwhm is not None:
                print(f"  → All new windows will get LSF FWHM = {lsf_fwhm} Å")
            elif resolution is not None:
                central_wave = (metadata['r_min'] + metadata['r_max']) / 2
                calculated_lsf = central_wave / resolution
                print(f"  → LSF FWHM will be calculated as λ_central/{resolution} = {calculated_lsf:.3f} Å")
        else:
            print("✓ Detected NEW format metadata - will propagate existing LSF FWHM")
            if 'wavelength_ranges' in metadata:
                print("  Existing LSF FWHM values:")
                for i, wr in enumerate(metadata['wavelength_ranges']):
                    if 'lsf_fwhm' in wr and wr['lsf_fwhm'] is not None:
                        central = (wr['r_min'] + wr['r_max']) / 2
                        print(f"    Range {i+1}: λ={central:.1f} Å, LSF FWHM={wr['lsf_fwhm']:.3f} Å")
        
        print()
        
    except Exception as e:
        print(f"Warning: Could not pre-check metadata format: {e}")
        print()
    
    # run re-processing pipeline with FIXED LSF FWHM handling
    try:
        results = process_existing_blobby_data(
            input_dir=input_dir,
            output_dir=output_dir,
            emission_lines=emission_lines,
            window_size=window_size,
            min_gap=min_gap,
            lsf_fwhm=lsf_fwhm,          # FIXED: properly propagated to windows
            resolution=resolution,       # alternative for LSF FWHM calculation
            dry_run=dry_run,
            create_plots=create_plots
        )
        
        print("\n" + "="*70)
        print("SUCCESS: Blobby3D Re-processing Complete with LSF FWHM")
        print("="*70)
        
        # print windowing summary with LSF FWHM info
        metadata = results['metadata']
        windows = results['windows']
        original_shape = results['original_data'].shape
        windowed_shape = results['windowed_data'].shape
        
        print(f"\nData reduction summary:")
        print(f"  Original: {original_shape} -> {original_shape[2]} wavelength bins")
        print(f"  Windowed: {windowed_shape} -> {windowed_shape[2]} wavelength bins")
        print(f"  Reduction factor: {original_shape[2]/windowed_shape[2]:.1f}×")
        print(f"  Coverage efficiency: {100*windowed_shape[2]/original_shape[2]:.1f}%")
        
        print(f"\nWindows created with LSF FWHM: {len(windows)}")
        lsf_values = []
        for i, window in enumerate(windows):
            actual_min = window.get('actual_r_min', window['r_min'])
            actual_max = window.get('actual_r_max', window['r_max'])
            n_bins = window.get('n_bins', 0)
            lsf_fwhm_val = window.get('lsf_fwhm', None)
            
            print(f"  {i+1}. {window['name']}: {actual_min:.1f}-{actual_max:.1f} Å ({n_bins} bins)")
            if lsf_fwhm_val is not None:
                print(f"      LSF FWHM: {lsf_fwhm_val:.3f} Å")
                lsf_values.append(lsf_fwhm_val)
            else:
                print(f"      LSF FWHM: Missing!")
        
        # Check if LSF FWHM values are properly assigned
        if len(lsf_values) == len(windows):
            print(f"\n SUCCESS: All {len(windows)} windows have LSF FWHM values!")
            if len(set(lsf_values)) == 1:
                print(f"   All windows use same LSF FWHM: {lsf_values[0]:.3f} Å")
            else:
                print(f"   LSF FWHM range: {min(lsf_values):.3f} - {max(lsf_values):.3f} Å")
        else:
            print(f"\n WARNING: {len(windows) - len(lsf_values)} windows missing LSF FWHM!")
        
        # show format information
        if results.get('format_converted', False):
            print(f"\nFormat conversion:")
            print(f"  ✓ Converted old format to new format with LSF FWHM")
            if results.get('conversion_lsf_fwhm'):
                print(f"  ✓ Applied LSF FWHM: {results['conversion_lsf_fwhm']:.3f} Å to all windows")
        else:
            print(f"\nMetadata format: {metadata.get('format', 'unknown')}")
            if metadata.get('format') == 'new':
                print(f"  ✓ Propagated per-window LSF FWHM from existing metadata")
        
        if not dry_run:
            print("\nOutput files ready for Blobby3D:")
            for name, path in results['written_files'].items():
                if path:  # some might be None
                    print(f"  - {path}")
            
            # Verify the output metadata contains LSF FWHM
            output_metadata_file = results['written_files']['metadata_file']
            print(f"\n Checking output metadata file...")
            try:
                with open(output_metadata_file, 'r') as f:
                    content = f.read()
                    wave_range_lines = [line for line in content.split('\n') if line.startswith('wave_range')]
                    
                print(f"Output metadata contains {len(wave_range_lines)} wave_range lines:")
                for line in wave_range_lines:
                    print(f"  {line}")
                    # Check if LSF FWHM is present (should have 6+ parts)
                    parts = line.split()
                    if len(parts) >= 7:  # wave_range + 5 numbers + LSF_FWHM
                        try:
                            lsf_val = float(parts[6])
                            print(f"    LSF FWHM = {lsf_val:.3f} Å")
                        except:
                            print(f"     LSF FWHM parse error")
                    else:
                        print(f"     Missing LSF FWHM (only {len(parts)} parts)")
                        
            except Exception as e:
                print(f"  Error checking metadata: {e}")
            
        
        if create_plots:
            print("\nDiagnostic plots created:")
            for name, path in results['plot_files'].items():
                print(f"  - {path}")
        
        
    except Exception as e:
        print(f"\nERROR: Re-processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



if __name__ == '__main__':
    main()
