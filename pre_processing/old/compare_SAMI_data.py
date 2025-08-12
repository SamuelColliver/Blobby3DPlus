#!/usr/bin/env python3
"""
Compare the raw SAMI data to the processed continuum data.
This script reads the SAMI blue and red integral field spectroscopy (IFS) data,
and compares it with the continuum data.
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

def read_data(filename, extension_name):
    """
    Read IFS data from FITS file for continuum.
    """
    print(f"Reading FITS file: {filename}")
    
    with fits.open(filename) as hdul:
        # read data
        data = hdul[extension_name].data
        header = hdul[extension_name].header

        # read dimensions from header
        naxis1 = header['NAXIS1']  # x dimension (expected)
        naxis2 = header['NAXIS2']  # y dimension (expected)
        naxis3 = header['NAXIS3']  # wavelength dimension (expected)
        
        print(f"Header dimensions: NAXIS1={naxis1}, NAXIS2={naxis2}, NAXIS3={naxis3}")
        print(f"Original FITS continuum shape: {data.shape}")
        
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
        data = transpose_data(data, "continuum")
    
    print(f"Final continuum shape: {data.shape}")
    
    return data

def main():
    """Main function for processing combined SAMI blue and red IFS data."""
    
    # +++++++++++++ Can be Edited +++++++++++++++++
    
    # input files
    input_file_blue = '/Users/scol0322/Documents/sami_data/65410_blue_7_Y13SAR1_P003_15T008_2013_03_05-2013_03_17.fits'
    blue_data_extension = 'PRIMARY'
    input_file_red = '/Users/scol0322/Documents/sami_data//65410_red_7_Y13SAR1_P003_15T008_2013_03_05-2013_03_17.fits'
    red_data_extension = 'PRIMARY'
    
    # continuum file
    continuum_file = '/Users/scol0322/Documents/sami_data/65410_7_Y13SAR1_P003_15T008_2013_03_05-2013_03_17_1_comp.fits'
    blue_extension_name_continuum = 'B_CONTINUUM'
    red_extension_name_continuum = 'R_CONTINUUM'

    blue_extension_name_line = 'B_LINE'
    red_extension_name_line = 'R_LINE'

    # +++++++++ End of Can be Edited +++++++++++++
    
    try:
        # BLUE Figure
        data = read_data(input_file_blue, blue_data_extension)
        lines = read_data(continuum_file, blue_extension_name_line)
        continuum = read_data(continuum_file, blue_extension_name_continuum)

        fig = plt.Figure()

        plt.plot(data[:,25,25], label = 'data')
        plt.plot(lines[:,25,25], label = "lines")
        plt.plot(continuum[:,25,25], label = 'continuum')
        plt.plot (lines[:,25,25] + continuum[:,25,25], label = 'lines + continuum')
        plt.plot (data[:,25,25] - continuum[:,25,25], label = 'data - continuum')
        plt.legend()
        plt.show()

        # RED Figure
        data = read_data(input_file_red, red_data_extension)
        lines = read_data(continuum_file, red_extension_name_line)
        continuum = read_data(continuum_file, red_extension_name_continuum)

        fig = plt.Figure()

        plt.plot(data[:,25,25], label = 'data')
        plt.plot(lines[:,25,25], label = "lines")
        plt.plot(continuum[:,25,25], label = 'continuum')
        plt.plot (lines[:,25,25] + continuum[:,25,25], label = 'lines + continuum')
        plt.plot (data[:,25,25] - continuum[:,25,25], label = 'data - continuum')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error processing SAMI data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()