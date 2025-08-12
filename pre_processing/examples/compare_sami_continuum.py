#!/usr/bin/env python3
"""
Example script for comparing raw SAMI data with processed continuum data.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io.fits_readers import read_sami_fits, read_continuum_fits


def plot_spectrum_comparison(data, lines, continuum, wavelengths, spaxel_coords, output_path=None):
    """
    Create comparison plot of data, lines, continuum, and combinations.
    
    Parameters
    ----------
    data : np.ndarray
        Raw data cube
    lines : np.ndarray  
        Line data cube
    continuum : np.ndarray
        Continuum data cube
    wavelengths : np.ndarray
        Wavelength array
    spaxel_coords : tuple
        (i, j) coordinates of spaxel to plot
    output_path : Path, optional
        Where to save the plot
    """
    i, j = spaxel_coords
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(wavelengths, data[:, i, j], label='Raw Data', linewidth=1)
    plt.plot(wavelengths, lines[:, i, j], label="Line Component", linewidth=1)
    plt.plot(wavelengths, continuum[:, i, j], label='Continuum', linewidth=1)
    plt.plot(wavelengths, lines[:, i, j] + continuum[:, i, j], 
             label='Lines + Continuum', linewidth=1, linestyle='--')
    plt.plot(wavelengths, data[:, i, j] - continuum[:, i, j], 
             label='Data - Continuum', linewidth=1, linestyle=':')
    
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title(f'SAMI Spectrum Comparison (spaxel {i}, {j})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved: {output_path}")
    
    plt.show()


def main():
    """Compare raw SAMI data with processed continuum data."""
    
    # +++++++++++++ Configuration - Edit This Section +++++++++++++++++
    
    # input files
    blue_data_file = '/Users/scol0322/Documents/sami_data/65410_blue_7_Y13SAR1_P003_15T008_2013_03_05-2013_03_17.fits'
    red_data_file = '/Users/scol0322/Documents/sami_data/65410_red_7_Y13SAR1_P003_15T008_2013_03_05-2013_03_17.fits'
    
    # continuum/processed file
    continuum_file = '/Users/scol0322/Documents/sami_data/65410_7_Y13SAR1_P003_15T008_2013_03_05-2013_03_17_1_comp.fits'
    
    # extensions
    blue_continuum_ext = 'B_CONTINUUM'
    red_continuum_ext = 'R_CONTINUUM'
    blue_line_ext = 'B_LINE'
    red_line_ext = 'R_LINE'
    
    # output directory for plots
    output_dir = '/Users/scol0322/Documents/output/sami_comparison'
    
    # spaxel to examine (will use centre if not specified)
    spaxel_coords = None  # (25, 25) or None for centre
    
    # +++++++++ End of Configuration +++++++++++++
    
    # validate input files
    files_to_check = [blue_data_file, red_data_file, continuum_file]
    for file_path in files_to_check:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    # create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("SAMI Data vs Continuum Comparison")
    print("=" * 40)
    print(f"Blue data: {Path(blue_data_file).name}")
    print(f"Red data: {Path(red_data_file).name}")
    print(f"Continuum file: {Path(continuum_file).name}")
    print()
    
    try:
        # process blue data
        print("Processing blue data...")
        blue_data, _, _, blue_header, blue_waves = read_sami_fits(blue_data_file)
        blue_lines = read_continuum_fits(continuum_file, blue_line_ext)
        blue_continuum = read_continuum_fits(continuum_file, blue_continuum_ext)
        
        # determine spaxel coordinates
        if spaxel_coords is None:
            centre_i, centre_j = blue_data.shape[1]//2, blue_data.shape[2]//2
            spaxel_coords = (centre_i, centre_j)
        
        print(f"Using spaxel coordinates: {spaxel_coords}")
        
        # create blue comparison plot
        blue_plot_path = output_path / 'blue_spectrum_comparison.png'
        plot_spectrum_comparison(
            blue_data, blue_lines, blue_continuum, blue_waves, 
            spaxel_coords, blue_plot_path
        )
        
        # process red data
        print("\nProcessing red data...")
        red_data, _, _, red_header, red_waves = read_sami_fits(red_data_file)
        red_lines = read_continuum_fits(continuum_file, red_line_ext)
        red_continuum = read_continuum_fits(continuum_file, red_continuum_ext)
        
        # create red comparison plot
        red_plot_path = output_path / 'red_spectrum_comparison.png'
        plot_spectrum_comparison(
            red_data, red_lines, red_continuum, red_waves,
            spaxel_coords, red_plot_path
        )
        
    except Exception as e:
        print(f"ERROR: Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()