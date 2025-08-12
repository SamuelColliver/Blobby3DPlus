#!/usr/bin/env python3
"""
SAMI IFS FITS Data Processor for Blobby3D
Processes FITS files containing SAMI IFS data with redshift correction,
continuum subtraction, and data cleaning.
Now includes weights and variance processing.
"""

import numpy as np
from astropy.io import fits
from astropy import constants as const
import sys
import warnings
from pathlib import Path

# suppress FITS warnings for cleaner output
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

def read_sami_fits(filename):
    """
    Read IFS data from FITS file including data, weights, and variance.
    Reads dimensions from header and handles the transpose correctly.
    """
    print(f"Reading FITS file: {filename}")
    
    with fits.open(filename) as hdul:
        print(f"Found {len(hdul)} extensions in FITS file")
        
        # print extension info for debugging
        for i, hdu in enumerate(hdul):
            ext_name = hdu.header.get('EXTNAME', f'Extension {i}')
            if hdu.data is not None:
                print(f"  Extension {i} ({ext_name}): {hdu.data.shape}")
            else:
                print(f"  Extension {i} ({ext_name}): No data")
        
        # find extensions with 3D data
        data_ext = None
        weights_ext = None
        var_ext = None
        
        for i, hdu in enumerate(hdul):
            if hdu.data is not None and len(hdu.data.shape) == 3:
                ext_name = hdu.header.get('EXTNAME', '').upper()
                
                # look for data extension (usually first 3D extension or named DATA)
                if data_ext is None and ('PRIMARY' in ext_name):
                    data_ext = i
                    print(f"  -> Found data in extension {i}")
                
                # look for weights extension
                elif 'WEIGHT' in ext_name:
                    weights_ext = i
                    print(f"  -> Found weights in extension {i}")
                
                # look for variance extension
                elif 'VARIANCE' in ext_name:
                    var_ext = i
                    print(f"  -> Found variance in extension {i}")
        
        # if data_ext is still None, use first 3D extension
        if data_ext is None:
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) == 3:
                    data_ext = i
                    print(f"  -> Using extension {i} as data (first 3D extension)")
                    break
        
        if data_ext is None:
            raise ValueError("No 3D data cube found in FITS file")
        
        # 4ead data
        data = hdul[data_ext].data
        header = hdul[data_ext].header
        
        # read weights if available
        weights = None
        if weights_ext is not None:
            weights = hdul[weights_ext].data
            print(f"  -> Read weights from extension {weights_ext}")
        
        # read variance if available
        variance = None
        if var_ext is not None:
            variance = hdul[var_ext].data
            print(f"  -> Read variance from extension {var_ext}")
        
        # if weights/variance weren't found by name, try to find them by position
        if weights is None or variance is None:
            print("  -> Searching for weights/variance by position...")
            extensions_3d = []
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) == 3:
                    extensions_3d.append(i)
            
            # common patterns: data, weights, variance OR data, variance, weights
            if len(extensions_3d) >= 3:
                if weights is None and len(extensions_3d) > 1:
                    weights_ext = extensions_3d[1]  # Second 3D extension
                    weights = hdul[weights_ext].data
                    print(f"  -> Assuming weights in extension {weights_ext}")
                
                if variance is None and len(extensions_3d) > 2:
                    var_ext = extensions_3d[2]  # Third 3D extension
                    variance = hdul[var_ext].data
                    print(f"  -> Assuming variance in extension {var_ext}")
        
        # read dimensions from header
        naxis1 = header['NAXIS1']  # X dimension (expected)
        naxis2 = header['NAXIS2']  # Y dimension (expected)
        naxis3 = header['NAXIS3']  # 2avelength dimension (expected)
        
        print(f"Header dimensions: NAXIS1={naxis1}, NAXIS2={naxis2}, NAXIS3={naxis3}")
        print(f"Original FITS data shape: {data.shape}")
        
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
                        pass  # Already correct
                    elif wavelength_axis == 1:
                        data_array = np.transpose(data_array, (1, 0, 2))
                    elif wavelength_axis == 2:
                        data_array = np.transpose(data_array, (2, 0, 1))
                
                print(f"  -> Transposed {label} to shape: {data_array.shape}")
            
            return data_array
        
        # transpose all arrays to correct orientation
        data = transpose_data(data, "data")
        weights = transpose_data(weights, "weights")
        variance = transpose_data(variance, "variance")
        
        # create wavelength array from header WCS
        # use CRVAL3, CDELT3, CRPIX3 for wavelength axis
        if 'CRVAL3' in header and 'CDELT3' in header and 'CRPIX3' in header:
            crval3 = header['CRVAL3']  # reference wavelength
            cdelt3 = header['CDELT3']  # wavelength increment  
            crpix3 = header['CRPIX3']  # reference pixel
            
            # create wavelength array
            pixels = np.arange(1, naxis3 + 1)  # FITS pixels start at 1
            wavelengths = crval3 + cdelt3 * (pixels - crpix3)
            
            print(f"Using CRVAL3/CDELT3/CRPIX3 wavelength solution:")
            print(f"  CRVAL3={crval3:.2f} Å, CDELT3={cdelt3:.4f} Å/pixel, CRPIX3={crpix3}")
            
        else:
            print("Warning: No standard wavelength WCS found in header")
            wavelengths = np.arange(naxis3, dtype=float)
    
    print(f"Final data shape: {data.shape}")
    if weights is not None:
        print(f"Final weights shape: {weights.shape}")
    if variance is not None:
        print(f"Final variance shape: {variance.shape}")
    print(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} Å")
    print(f"Number of wavelength channels: {len(wavelengths)}")
    
    return data, weights, variance, header, wavelengths

def redshift_correct_wavelengths(wavelengths, redshift):
    """
    Apply redshift correction to wavelengths.
    """
    if redshift == 0:
        return wavelengths
    
    rest_wavelengths = wavelengths / (1 + redshift)
    print(f"Rest-frame wavelength range: {rest_wavelengths[0]:.2f} - {rest_wavelengths[-1]:.2f} Å")
    
    return rest_wavelengths

def subtract_continuum(data, weights=None):
    """
    Subtract continuum using nanmedian across wavelength axis.
    For weights, continuum subtraction doesn't apply - weights remain unchanged.
    """
    print("Subtracting continuum using nanmedian...")
    
    # calculate continuum (median along wavelength axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        continuum = np.nanmedian(data, axis=0, keepdims=True)
    
    # replace NaN continuum values with 0 to avoid issues
    continuum = np.nan_to_num(continuum, nan=0.0)
    
    # subtract continuum from data
    continuum_subtracted = data - continuum
    
    return continuum_subtracted

def clean_data(data, pixelscale_arcsec, weights=None, variance=None):
    """
    Remove rows and columns from edges where all wavelengths are NaN,
    effectively cropping the data to its valid bounds.
    Apply the same cropping to weights and variance.
    Calculate coordinate ranges based on pixel scale with centre at 0.0.
    """
    print("Cleaning data - cropping NaN edges...")
    
    # store original dimensions for coordinate calculation
    original_height, original_width = data.shape[-2], data.shape[-1]
    
    # for 3D data (wavelength, y, x), check if all wavelengths are NaN for each spatial position
    if data.ndim == 3:
        # check for positions where all wavelengths are NaN
        all_nan_spatial = np.all(np.isnan(data), axis=0)  # Shape: (y, x)
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
    # x-coordinates: left edge moves right by removed pixels * scale
    new_x_left = original_x_range[0] + (cols_removed_left * pixelscale_arcsec)
    new_x_right = original_x_range[1] - (cols_removed_right * pixelscale_arcsec)
    
    # y-coordinates: bottom edge moves up by removed pixels * scale  
    # (assuming y increases upward, row 0 is bottom)
    new_y_bottom = original_y_range[0] + (rows_removed_bottom * pixelscale_arcsec)
    new_y_top = original_y_range[1] - (rows_removed_top * pixelscale_arcsec)
    
    cropped_x_range = (new_x_left, new_x_right)
    cropped_y_range = (new_y_bottom, new_y_top)
    
    # crop the data
    if data.ndim == 3:
        cropped_data = data[:, row_start:row_end, col_start:col_end]
        # apply same cropping to weights and variance
        cropped_weights = weights[:, row_start:row_end, col_start:col_end] if weights is not None else None
        cropped_variance = variance[:, row_start:row_end, col_start:col_end] if variance is not None else None
        
        # update the valid mask to match the cropped dimensions
        valid_mask = ~np.all(np.isnan(cropped_data), axis=0)
    else:
        cropped_data = data[row_start:row_end, col_start:col_end]
        cropped_weights = weights[row_start:row_end, col_start:col_end] if weights is not None else None
        cropped_variance = variance[row_start:row_end, col_start:col_end] if variance is not None else None
        valid_mask = ~np.isnan(cropped_data)
    
    # calculate statistics
    original_size = original_height * original_width
    cropped_size = cropped_data.shape[-2] * cropped_data.shape[-1]
    removed_count = original_size - cropped_size
    
    # create coordinate info dictionary
    coord_info = {
        'x_min': new_x_left,
        'x_max': new_x_right,
        'y_min': new_y_bottom,
        'y_max': new_y_top,
        'spatial_sampling': pixelscale_arcsec
    }
    
    print(f"Original dimensions: {data.shape}")
    print(f"Cropped dimensions: {cropped_data.shape}")
    if cropped_weights is not None:
        print(f"Cropped weights dimensions: {cropped_weights.shape}")
    if cropped_variance is not None:
        print(f"Cropped variance dimensions: {cropped_variance.shape}")
    print(f"Cropped from (rows {row_start}:{row_end}, cols {col_start}:{col_end})")
    print(f"Removed {removed_count} edge spaxels out of {original_size} total")
    print(f"Pixel scale: {pixelscale_arcsec} arcsec/pixel")
    print(f"Original coordinate ranges:")
    print(f"  X: {original_x_range[0]:.2f} to {original_x_range[1]:.2f} arcsec")
    print(f"  Y: {original_y_range[0]:.2f} to {original_y_range[1]:.2f} arcsec")
    print(f"Cropped coordinate ranges:")
    print(f"  X: {cropped_x_range[0]:.2f} to {cropped_x_range[1]:.2f} arcsec")
    print(f"  Y: {cropped_y_range[0]:.2f} to {cropped_y_range[1]:.2f} arcsec")
    print(f"Pixels removed: {rows_removed_top} top, {rows_removed_bottom} bottom, {cols_removed_left} left, {cols_removed_right} right")
    
    return cropped_data, cropped_weights, cropped_variance, valid_mask, coord_info

def save_data_and_metadata(data, weights, variance, wavelengths, coord_info, output_dir="./"):
    """
    Save data, weights, variance and metadata in Blobby3D format.
    
    Parameters:
    -----------
    data : numpy.ndarray
        3D data cube (wavelength, y, x)
    weights : numpy.ndarray or None
        3D weights cube (wavelength, y, x)
    variance : numpy.ndarray or None
        3D variance cube (wavelength, y, x)
    wavelengths : numpy.ndarray
        Wavelength array
    output_dir : str
        Output directory path
    """
    print("Saving data and metadata...")
    
    # Create output directory and file paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_filename = output_path / 'data.txt'
    weights_filename = output_path / 'weights.txt'
    variance_filename = output_path / 'var.txt'
    metadata_filename = output_path / 'metadata.txt'

    # Convert NaN to zero for data
    data_clean = np.nan_to_num(data, nan=0.0)

    # Reshape data for output (spaxels × wavelengths)
    data_clean_output = np.zeros((data_clean.shape[1] * data_clean.shape[2], data_clean.shape[0]))

    i = 0
    for y_idx in range(data_clean.shape[1]):
        for x_idx in range(data_clean.shape[2]):
            data_clean_output[i, :] = data_clean[:, y_idx, x_idx]
            i += 1

    # Save data
    np.savetxt(data_filename, data_clean_output, fmt='%.6e')
    print(f"Saved data to {data_filename}")
    print(f"Data shape: {data_clean_output.shape[0]} spaxels × {data_clean_output.shape[1]} wavelengths")
    
    # Save weights if available
    if weights is not None:
        weights_clean = np.nan_to_num(weights, nan=0.0)
        weights_clean_output = np.zeros((weights_clean.shape[1] * weights_clean.shape[2], weights_clean.shape[0]))
        
        i = 0
        for y_idx in range(weights_clean.shape[1]):
            for x_idx in range(weights_clean.shape[2]):
                weights_clean_output[i, :] = weights_clean[:, y_idx, x_idx]
                i += 1
                
        np.savetxt(weights_filename, weights_clean_output, fmt='%.6e')
        print(f"Saved weights to {weights_filename}")
        print(f"Weights shape: {weights_clean_output.shape[0]} spaxels × {weights_clean_output.shape[1]} wavelengths")
    else:
        print("No weights data to save")
    
    # Save variance if available
    if variance is not None:
        variance_clean = np.nan_to_num(variance, nan=0.0)
        variance_clean_output = np.zeros((variance_clean.shape[1] * variance_clean.shape[2], variance_clean.shape[0]))
        
        i = 0
        for y_idx in range(variance_clean.shape[1]):
            for x_idx in range(variance_clean.shape[2]):
                variance_clean_output[i, :] = variance_clean[:, y_idx, x_idx]
                i += 1
                
        np.savetxt(variance_filename, variance_clean_output, fmt='%.6e')
        print(f"Saved variance to {variance_filename}")
        print(f"Variance shape: {variance_clean_output.shape[0]} spaxels × {variance_clean_output.shape[1]} wavelengths")
    else:
        print("No variance data to save")
    
    # Wavelength bin edges
    if len(wavelengths) > 1:
        wave_bin_width = np.median(np.diff(wavelengths))
        wave_min = np.min(wavelengths) - wave_bin_width / 2
        wave_max = np.max(wavelengths) + wave_bin_width / 2
    else:
        wave_min = wave_max = wavelengths[0]
        
    # write metadata file
    with open(metadata_filename, 'w') as f:
        f.write("# Blobby3D Metadata File\n")
        f.write(f"ni {data_clean.shape[1]}\n")
        f.write(f"nj {data_clean.shape[2]}\n")
        f.write(f"x_min {coord_info['x_min']}\n")
        f.write(f"x_max {coord_info['x_max']}\n")
        f.write(f"y_min {coord_info['y_min']}\n")
        f.write(f"y_max {coord_info['y_max']}\n")
        #save as wave_range {min wavelength angstrom} {max wavelength angstrom} {first index} {last index} {total index}
        f.write(f"wave_range {wave_min:.6f} {wave_max:.6f} {0} {data_clean.shape[0]-1} {data_clean.shape[0]}\n")

    print(f"Saved metadata to {metadata_filename}")
    
    # prepare return filenames
    output_files = [str(data_filename), str(metadata_filename)]
    if weights is not None:
        output_files.append(str(weights_filename))
    if variance is not None:
        output_files.append(str(variance_filename))
    
    return output_files

def main():
    """Main function for processing SAMI IFS FITS data."""
    
    # +++++++++++++ Can be Edited +++++++++++++++++
    input_file = '../65410_red_7_Y13SAR1_P003_15T008.fits'   # Path to input SAMI FITS file
    input_file_blue = '../65410_blue_7_Y13SAR1_P003_15T008.fits'   # Path to input SAMI FITS file
    output_dir = './examples/65410/'         # Output directory
    pix_size_arcsec = 0.5                                 # SAMI spatial sampling is at 0.5 arcsec
    redshift = 0.043514                                   # Redshift for correction (z)
    verbose = True                                        # Set to False for minimal output
    
    # Optional: Override coordinate system if FITS header is incomplete
    # Set to None to use header values
    manual_coords = None  # Example: {'x_pixel_size': 0.5, 'y_pixel_size': 0.5}
    
    # Data quality settings
    remove_invalid_spaxels = True           # Remove spaxels with all NaN
    continuum_method = 'nanmedian'          # Method for continuum subtraction
    # +++++++++ End of Can be Edited +++++++++++++
    
    # Convert to Path objects for better handling
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    # Check input file exists
    if not input_path.exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("SAMI IFS FITS Data Processor for Blobby3D")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Redshift: {redshift}")
    print(f"Continuum method: {continuum_method}")
    print(f"Remove invalid spaxels: {remove_invalid_spaxels}")
    print()
    
    try:
        # Read SAMI FITS data including weights and variance
        print("Step 1: Reading SAMI FITS file...")
        data, weights, variance, header, wavelengths = read_sami_fits(input_file)
        
        if verbose:
            print(f"  - Data cube shape: {data.shape}")
            print(f"  - Weights available: {'Yes' if weights is not None else 'No'}")
            print(f"  - Variance available: {'Yes' if variance is not None else 'No'}")
            print(f"  - Instrument: {header.get('INSTRUME', 'Unknown')}")
            print(f"  - Plate ID: {header.get('PLATEID', 'Unknown')}")
            print(f"  - Grating: {header.get('GRATID', 'Unknown')}")
            print(f"  - Data units: {header.get('BUNIT', 'Unknown')}")
            print(f"  - Original wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} Å")
            print()
        
        # Apply redshift correction
        if redshift != 0.0:
            print("Step 2: Applying redshift correction...")
            rest_wavelengths = redshift_correct_wavelengths(wavelengths, redshift)
            if verbose:
                print(f"  - Rest-frame wavelength range: {rest_wavelengths[0]:.2f} - {rest_wavelengths[-1]:.2f} Å")
                print()
        else:
            print("Step 2: No redshift correction applied (z=0)")
            rest_wavelengths = wavelengths
            if verbose:
                print()
        
        # Subtract continuum (only from data, not weights/variance)
        print("Step 3: Subtracting continuum...")
        continuum_subtracted = subtract_continuum(data, weights)
        if verbose:
            print(f"  - Using {continuum_method} across wavelength axis")
            print(f"  - Data range after continuum subtraction: {np.nanmin(continuum_subtracted):.2e} to {np.nanmax(continuum_subtracted):.2e}")
            print()
        
        # Clean data (crop all arrays consistently)
        if remove_invalid_spaxels:
            print("Step 4: Cleaning data...")
            cleaned_data, cleaned_weights, cleaned_variance, valid_mask, coord_info = clean_data(
                continuum_subtracted, pix_size_arcsec, weights, variance)
            if verbose:
                total_spaxels = data.shape[1] * data.shape[2]
                valid_spaxels = np.sum(valid_mask)
                print(f"  - Valid spaxels: {valid_spaxels}/{total_spaxels} ({100*valid_spaxels/total_spaxels:.1f}%)")
                print()
        else:
            print("Step 4: Skipping data cleaning (keeping all spaxels)")
            cleaned_data = continuum_subtracted
            cleaned_weights = weights
            cleaned_variance = variance
            height, width = clean_data.shape[-2], clean_data.shape[-1]
            coord_info = {
                'x_min': -(width * pix_size_arcsec) / 2.0,
                'x_max': (width * pix_size_arcsec) / 2.0,
                'y_min': -(height * pix_size_arcsec) / 2.0,
                'y_max': (height * pix_size_arcsec) / 2.0,
                'spatial_sampling': pix_size_arcsec
            }
            valid_mask = np.ones((data.shape[1], data.shape[2]), dtype=bool)
            if verbose:
                print()
        
        # Save results
        print("Step 5: Saving data and metadata...")
        output_files = save_data_and_metadata(
            cleaned_data, cleaned_weights, cleaned_variance, rest_wavelengths, coord_info, output_dir)
        
        print()
        print("=" * 50)
        print("SAMI data processing complete!")
        print(f"Output files:")
        for filename in output_files:
            print(f"  - {filename}")
        
        if verbose:
            print()
            print("Files ready for Blobby3D analysis.")
            print("Data format: rows = spaxels, columns = wavelengths")
            print("Metadata contains bin edges for proper spatial/spectral binning")
            
            # print some final statistics
            final_data = np.loadtxt(output_files[0])  # data.txt
            print(f"Final data shape: {final_data.shape[0]} spaxels × {final_data.shape[1]} wavelengths")
            print(f"Data range: {np.min(final_data):.2e} to {np.max(final_data):.2e}")
            
            if cleaned_weights is not None:
                final_weights = np.loadtxt(output_files[2])  # weights.txt
                print(f"Weights range: {np.min(final_weights):.2e} to {np.max(final_weights):.2e}")
            
            if cleaned_variance is not None:
                var_file_idx = 3 if cleaned_weights is not None else 2
                final_variance = np.loadtxt(output_files[var_file_idx])  # variance.txt
                print(f"Variance range: {np.min(final_variance):.2e} to {np.max(final_variance):.2e}")
        
    except Exception as e:
        print(f"Error processing SAMI data: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()