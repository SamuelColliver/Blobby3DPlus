#!/usr/bin/env python3
"""
FITS file readers for different IFS instruments.
"""

import numpy as np
from astropy.io import fits
import warnings
from pathlib import Path

# suppress FITS warnings for cleaner output
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)


def transpose_data_to_standard(data_array, naxis1, naxis2, naxis3, label="data"):
    """
    Transpose data array to standard (wavelength, y, x) format.
    
    Parameters
    ----------
    data_array : np.ndarray or None
        Data array to transpose
    naxis1, naxis2, naxis3 : int
        FITS header dimensions
    label : str
        Label for logging
    
    Returns
    -------
    np.ndarray or None
        Transposed array in (wavelength, y, x) format
    """
    if data_array is None:
        return None
        
    # FITS standard is (NAXIS3, NAXIS2, NAXIS1) = (wavelength, y, x)
    expected_shape = (naxis3, naxis2, naxis1)
    
    if data_array.shape == expected_shape:
        return data_array
        
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


def create_wavelength_array(header, naxis3):
    """
    Create wavelength array from FITS header WCS information.
    
    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header containing WCS information
    naxis3 : int
        Number of wavelength channels
    
    Returns
    -------
    np.ndarray
        Wavelength array in Angstroms
    """
    if 'CRVAL3' in header and 'CDELT3' in header and 'CRPIX3' in header:
        crval3 = header['CRVAL3']  # reference wavelength
        cdelt3 = header['CDELT3']  # wavelength increment  
        crpix3 = header['CRPIX3']  # reference pixel
        
        # create wavelength array
        pixels = np.arange(1, naxis3 + 1)  # FITS pixels start at 1
        wavelengths = crval3 + cdelt3 * (pixels - crpix3)
        
        print(f"Using CRVAL3/CDELT3/CRPIX3 wavelength solution:")
        print(f"  CRVAL3={crval3:.2f} Å, CDELT3={cdelt3:.4f} Å/pixel, CRPIX3={crpix3}")
        
        return wavelengths
    else:
        print("Warning: No standard wavelength WCS found in header")
        return np.arange(naxis3, dtype=float)


def read_sami_fits(filename):
    """
    Read SAMI IFS data from FITS file including data, weights, and variance.
    
    Parameters
    ----------
    filename : str or Path
        Path to SAMI FITS file
    
    Returns
    -------
    tuple
        (data, weights, variance, header, wavelengths)
        - data: 3D array (wavelength, y, x)
        - weights: 3D array (wavelength, y, x) or None
        - variance: 3D array (wavelength, y, x) or None
        - header: FITS header
        - wavelengths: 1D array of wavelengths in Angstroms
    """
    print(f"Reading SAMI FITS file: {filename}")
    
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
        
        # read data
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
                    weights_ext = extensions_3d[1]  # second 3D extension
                    weights = hdul[weights_ext].data
                    print(f"  -> Assuming weights in extension {weights_ext}")
                
                if variance is None and len(extensions_3d) > 2:
                    var_ext = extensions_3d[2]  # third 3D extension
                    variance = hdul[var_ext].data
                    print(f"  -> Assuming variance in extension {var_ext}")
        
        # read dimensions from header
        naxis1 = header['NAXIS1']  # x dimension
        naxis2 = header['NAXIS2']  # y dimension
        naxis3 = header['NAXIS3']  # wavelength dimension
        
        print(f"Header dimensions: NAXIS1={naxis1}, NAXIS2={naxis2}, NAXIS3={naxis3}")
        print(f"Original FITS data shape: {data.shape}")
        
        # transpose all arrays to correct orientation
        data = transpose_data_to_standard(data, naxis1, naxis2, naxis3, "data")
        weights = transpose_data_to_standard(weights, naxis1, naxis2, naxis3, "weights")
        variance = transpose_data_to_standard(variance, naxis1, naxis2, naxis3, "variance")
        
        # create wavelength array from header WCS
        wavelengths = create_wavelength_array(header, naxis3)
    
    print(f"Final data shape: {data.shape}")
    if weights is not None:
        print(f"Final weights shape: {weights.shape}")
    if variance is not None:
        print(f"Final variance shape: {variance.shape}")
    print(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} Å")
    print(f"Number of wavelength channels: {len(wavelengths)}")
    
    return data, weights, variance, header, wavelengths


def read_continuum_fits(filename, extension_name):
    """
    Read continuum data from FITS file.
    
    Parameters
    ----------
    filename : str or Path
        Path to continuum FITS file
    extension_name : str
        Name of the extension containing continuum data
    
    Returns
    -------
    np.ndarray
        Continuum data in (wavelength, y, x) format
    """
    print(f"Reading continuum from FITS file: {filename}")
    print(f"Extension: {extension_name}")
    
    with fits.open(filename) as hdul:
        # read data
        continuum = hdul[extension_name].data
        header = hdul[extension_name].header

        # read dimensions from header
        naxis1 = header['NAXIS1']  # x dimension
        naxis2 = header['NAXIS2']  # y dimension
        naxis3 = header['NAXIS3']  # wavelength dimension
        
        print(f"Header dimensions: NAXIS1={naxis1}, NAXIS2={naxis2}, NAXIS3={naxis3}")
        print(f"Original continuum shape: {continuum.shape}")
        
        # transpose to correct orientation
        continuum = transpose_data_to_standard(continuum, naxis1, naxis2, naxis3, "continuum")
    
    print(f"Final continuum shape: {continuum.shape}")
    return continuum


def read_generic_ifs_fits(filename, data_extension='PRIMARY', wavelength_extension=None):
    """
    Generic reader for IFS FITS files from other instruments.
    
    Parameters
    ----------
    filename : str or Path
        Path to FITS file
    data_extension : str
        Name of extension containing the data cube
    wavelength_extension : str, optional
        Name of extension containing wavelength information
    
    Returns
    -------
    tuple
        (data, weights, variance, header, wavelengths)
        Similar to read_sami_fits but may have None for unavailable arrays
    """
    print(f"Reading generic IFS FITS file: {filename}")
    
    with fits.open(filename) as hdul:
        # read main data
        data = hdul[data_extension].data
        header = hdul[data_extension].header
        
        # get dimensions
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2'] 
        naxis3 = header['NAXIS3']
        
        # transpose data
        data = transpose_data_to_standard(data, naxis1, naxis2, naxis3, "data")
        
        # create wavelength array
        if wavelength_extension and wavelength_extension in [hdu.header.get('EXTNAME', '') for hdu in hdul]:
            # read wavelengths from separate extension
            wavelengths = hdul[wavelength_extension].data
            print(f"Read wavelengths from extension {wavelength_extension}")
        else:
            # try to create from header
            wavelengths = create_wavelength_array(header, naxis3)
        
        # for now, no weights or variance for generic files
        weights = None
        variance = None
        
        print(f"Generic IFS data shape: {data.shape}")
        print(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} Å")
        
        return data, weights, variance, header, wavelengths