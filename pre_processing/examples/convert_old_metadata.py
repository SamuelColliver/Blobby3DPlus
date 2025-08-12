#!/usr/bin/env python3
"""
Example script for converting old format Blobby3D metadata to new format with LSF FWHM.
"""

import sys
from pathlib import Path

# add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io.blobby_io import convert_old_to_new_metadata, load_blobby_metadata


def main():
    """Convert old format metadata files to new format with LSF FWHM."""
    
    # +++++++++++++ Configuration - Edit This Section +++++++++++++++++
    
    # input metadata file (old format)
    old_metadata_file = '../../examples/485885-old-meta/metadata.txt'
    
    # output metadata file (new format) - None for auto-naming
    new_metadata_file = None  # will create 'metadata_new.txt'
    
    # LSF FWHM specification (choose ONE of these options):
    
    # Option 1: Provide LSF FWHM directly (in Angstroms)
    lsf_fwhm = 1.61             # use this if you know the LSF FWHM value
    resolution = None          # set to None when using lsf_fwhm
    
    # Option 2: Calculate LSF FWHM from spectral resolution
    # lsf_fwhm = None          # set to None when using resolution
    # resolution = 4900        # R = λ/Δλ, will calculate LSF FWHM = λ_central/R
    
    # +++++++++ End of Configuration +++++++++++++
    
    # validate input file
    old_path = Path(old_metadata_file)
    if not old_path.exists():
        print(f"Error: Input metadata file not found: {old_metadata_file}")
        sys.exit(1)
    
    print("Blobby3D Metadata Format Converter (with LSF FWHM)")
    print("=" * 60)
    print(f"Input file: {old_metadata_file}")
    
    if lsf_fwhm is not None:
        print(f"LSF FWHM: {lsf_fwhm} Å (provided directly)")
    elif resolution is not None:
        print(f"Resolution: R = {resolution} (will calculate LSF FWHM)")
    else:
        print("Warning: No LSF FWHM or resolution provided")
    print()
    
    try:
        # first, check what format the input file is
        print("Checking input file format...")
        metadata = load_blobby_metadata(old_metadata_file)
        
        if metadata.get('format') == 'new':
            print("Input file is already in new format!")
            print("No conversion needed.")
            
            # show existing LSF FWHM info if available
            if 'wavelength_ranges' in metadata:
                print("\nExisting LSF FWHM information:")
                for i, wr in enumerate(metadata['wavelength_ranges']):
                    if 'lsf_fwhm' in wr and wr['lsf_fwhm'] is not None:
                        central_wave = (wr['r_min'] + wr['r_max']) / 2
                        print(f"  Range {i+1}: λ={central_wave:.1f} Å, LSF FWHM={wr['lsf_fwhm']:.3f} Å")
                    else:
                        print(f"  Range {i+1}: No LSF FWHM available")
            return
        
        # show what the old format looks like
        print(f"\nOld format detected:")
        with open(old_path, 'r') as f:
            old_content = f.read().strip()
            print(f"  {old_content}")
        
        # calculate what LSF FWHM will be used
        if lsf_fwhm is not None:
            print(f"\nUsing provided LSF FWHM: {lsf_fwhm} Å")
        elif resolution is not None:
            # calculate central wavelength for preview
            central_wave = (metadata['r_min'] + metadata['r_max']) / 2
            calculated_lsf = central_wave / resolution
            print(f"\nCalculating LSF FWHM from resolution:")
            print(f"  Central wavelength: {central_wave:.1f} Å")
            print(f"  Resolution: R = {resolution}")
            print(f"  LSF FWHM = {central_wave:.1f}/{resolution} = {calculated_lsf:.3f} Å")
        
        # convert to new format
        print(f"\nConverting old format to new format with LSF FWHM...")
        new_path = convert_old_to_new_metadata(
            old_metadata_file, 
            new_metadata_file,
            lsf_fwhm=lsf_fwhm,
            resolution=resolution
        )
        
        print(f"\n{'='*60}")
        print("CONVERSION COMPLETE")
        print(f"{'='*60}")
        
        print(f"Original file: {old_path}")
        print(f"New file: {new_path}")
        
        # show the new format
        print(f"\nNew format with LSF FWHM:")
        with open(new_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line_num <= 20:  # show first 20 lines
                    print(f"  {line.rstrip()}")
                elif line_num == 21:
                    print(f"  ... (file continues)")
                    break
        
        # verify the new file can be loaded
        print(f"\nVerifying new file...")
        new_metadata = load_blobby_metadata(new_path)
        print(f"  ✓ New file loads successfully")
        print(f"  ✓ Dimensions: {new_metadata['ni']} x {new_metadata['nj']} x {new_metadata['total_bins']}")
        print(f"  ✓ Wavelength range: {new_metadata['r_min']:.2f} - {new_metadata['r_max']:.2f} Å")
        
        # show LSF FWHM information
        if 'wavelength_ranges' in new_metadata:
            print(f"  ✓ LSF FWHM information:")
            for i, wr in enumerate(new_metadata['wavelength_ranges']):
                if 'lsf_fwhm' in wr and wr['lsf_fwhm'] is not None:
                    central_wave = (wr['r_min'] + wr['r_max']) / 2
                    print(f"      Range {i+1}: λ={central_wave:.1f} Å, LSF FWHM={wr['lsf_fwhm']:.3f} Å")
        
        print(f"\nConversion successful!")
        print(f"You can now use '{new_path}' with the new processing pipeline.")
        print(f"The new format includes LSF FWHM for accurate spectral analysis.")
        
    except Exception as e:
        print(f"\nERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def batch_convert_metadata(input_dir, output_dir=None, lsf_fwhm=None, resolution=None):
    """
    Convert all metadata.txt files in a directory from old to new format with LSF FWHM.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing subdirectories with metadata.txt files
    output_dir : str or Path, optional
        Output directory. If None, creates converted files in same locations.
    lsf_fwhm : float, optional
        LSF FWHM in Angstroms to add to converted files
    resolution : float, optional
        Spectral resolution R = λ/Δλ to calculate LSF FWHM
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    print(f"Batch converting metadata files to new format with LSF FWHM")
    print(f"Input directory: {input_dir}")
    if lsf_fwhm is not None:
        print(f"LSF FWHM: {lsf_fwhm} Å (fixed value)")
    elif resolution is not None:
        print(f"Resolution: R = {resolution} (will calculate per-file LSF FWHM)")
    print("=" * 60)
    
    # find all metadata.txt files
    metadata_files = list(input_path.rglob('metadata.txt'))
    
    if not metadata_files:
        print("No metadata.txt files found!")
        return
    
    print(f"Found {len(metadata_files)} metadata files:")
    for i, file_path in enumerate(metadata_files, 1):
        rel_path = file_path.relative_to(input_path)
        print(f"  {i}. {rel_path}")
    
    print()
    
    converted_count = 0
    already_new_count = 0
    failed_count = 0
    
    for file_path in metadata_files:
        rel_path = file_path.relative_to(input_path)
        print(f"Processing: {rel_path}")
        
        try:
            # check format
            metadata = load_blobby_metadata(file_path)
            
            if metadata.get('format') == 'new':
                print(f"  Already in new format - skipping")
                already_new_count += 1
                continue
            
            # determine output path
            if output_dir:
                output_path = Path(output_dir) / rel_path.parent / f"{rel_path.stem}_new{rel_path.suffix}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path = file_path.parent / f"{file_path.stem}_new{file_path.suffix}"
            
            # convert
            convert_old_to_new_metadata(
                file_path, 
                output_path, 
                lsf_fwhm=lsf_fwhm, 
                resolution=resolution
            )
            print(f"  ✓ Converted -> {output_path.name}")
            
            if lsf_fwhm is not None:
                print(f"    Added LSF FWHM: {lsf_fwhm} Å")
            elif resolution is not None:
                # calculate LSF FWHM for this file
                central_wave = (metadata['r_min'] + metadata['r_max']) / 2
                calculated_lsf = central_wave / resolution
                print(f"    Calculated LSF FWHM: {calculated_lsf:.3f} Å (λ={central_wave:.1f} Å, R={resolution})")
            
            converted_count += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_count += 1
    
    print(f"\n{'='*60}")
    print("BATCH CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(metadata_files)}")
    print(f"Converted: {converted_count}")
    print(f"Already new format: {already_new_count}")
    print(f"Failed: {failed_count}")
    
    if converted_count > 0:
        print(f"\nAll converted files now include LSF FWHM information!")
        print(f"Use the new format files with the updated processing pipeline.")


if __name__ == '__main__':
    # check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch':
            if len(sys.argv) > 2:
                input_dir = sys.argv[2]
                output_dir = sys.argv[3] if len(sys.argv) > 3 else None
                
                # parse LSF FWHM or resolution
                lsf_fwhm = None
                resolution = None
                if len(sys.argv) > 4:
                    value = float(sys.argv[4])
                    if len(sys.argv) > 5 and sys.argv[5].lower() in ['lsf', 'lsf_fwhm']:
                        lsf_fwhm = value
                    else:
                        resolution = value  # default to resolution
                
                batch_convert_metadata(input_dir, output_dir, lsf_fwhm, resolution)
            else:
                print("Usage: python convert_old_metadata.py --batch <input_dir> [output_dir] [value] [lsf|resolution]")
                print("Examples:")
                print("  python convert_old_metadata.py --batch /data/blobby/")
                print("  python convert_old_metadata.py --batch /data/blobby/ /output/ 1.8 lsf")
        else:
            # single file conversion
            old_file = sys.argv[1]
            new_file = sys.argv[2] if len(sys.argv) > 2 else None
            
            # parse LSF FWHM or resolution
            lsf_fwhm = None
            resolution = None
            if len(sys.argv) > 3:
                value = float(sys.argv[3])
                if len(sys.argv) > 4 and sys.argv[4].lower() in ['lsf', 'lsf_fwhm']:
                    lsf_fwhm = value
                else:
                    resolution = value
            
            try:
                result = convert_old_to_new_metadata(old_file, new_file, lsf_fwhm=lsf_fwhm, resolution=resolution)
                print(f"Converted: {old_file} -> {result}")
                if lsf_fwhm is not None:
                    print(f"Added LSF FWHM: {lsf_fwhm} Å")
                elif resolution is not None:
                    print(f"Added resolution: R = {resolution}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        # run the main configuration-based conversion
        main()
    