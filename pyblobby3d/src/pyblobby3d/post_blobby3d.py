"""Post analysis for Blobby3D output.

Updated to support multiple wavelength windows.

@original author: Mathew Varidel
@edited by: Samuel Colliver

"""

from pathlib import Path
import numpy as np
import pandas as pd

from .meta import Metadata


class PostBlobby3D:

    def __init__(
            self, samples_path, data_path, var_path, metadata_path,
            save_maps=True, save_precon=True, save_con=True,
            nlines=1, nsigmad=2):
        """Blobby3D postprocess object.

        This provides a read and storage object for analysis of multi-window data.

        Parameters
        ----------
        samples_path : str or pathlib object
            Sample path. This can be either DNest4 samples or posterior
            samples.
        data_path : str or pathlib object
            Data cube path. Data cubes are assumed to be in whitespace
            separated text files. The values should be in row-major order.
        var_path : str or pathlib object
            Variance cube path. Variance cubes are assumed to be in whitespace
            separated text files. The values should be in row-major order.
        metadata_path : str or pathlib object
            Metadata file path. This path records the coordinates for your
            cube and supports multiple wavelength windows.
        save_maps : bool, optional
            Maps were saved to sampled by Blobby3D. The default is True.
        save_precon : bool, optional
            Preconvolved cubes were saved to samples by Blobby3D. The default
            is True.
        save_con : bool, optional
            Convolved cubes were saved to samples by Blobby3D. The default is
            True.
        nlines : int, optional
            Number of lines modelled during run. A coupled line count as one.
            Therefore modelling H-alpha and [NII] at 6548.1 A and 6583.1 A
            with a fixed ratio is counted as two lines in total. The default is
            1.
        nsigmad : int, optional
            The degree to which white and shot noise are modelled. The default
            is 2.

        Attributes
        ----------
        nsamples : int
            Number of samples recorded.
        max_blobs : int
            Maximum number of blobs allowed in run.
        metadata : Metadata object
            See Metadata class docstring for details.
        data : np.ndarray
        var : np.ndarray
        maps : np.ndarray
            2D maps of flux per line, velocity, and velocity dispersion for
            each sample.
        precon_cubes : np.ndarray
            Preconvolved model cubes for each sample.
        con_cubes : np.ndarray
            Convolved model cubes for each sample.
        global_param : pd.DataFrame
            Global parameters.
        blob_param : pd.DataFrame
            Parameters for each blob.
        """
        self._posterior_path = Path(samples_path)
        self._data_path = Path(data_path)
        self._var_path = Path(var_path)
        self._metadata_path = Path(metadata_path)
        self._nlines = nlines
        self._nsigmad = nsigmad

        # Import metadata (supports both old and new formats)
        self.metadata = Metadata(self._metadata_path)
        
        # Load data cubes
        self.data = self._read_cube(data_path)
        print("Image Loaded...")

        self.var = self._read_cube(var_path)
        print("Variance Loaded...")

        # Posterior samples
        samples = np.atleast_2d(np.loadtxt(samples_path))
        self.nsamples = samples.shape[0]

        # Process samples based on what was saved
        self._process_samples(samples, save_maps, save_precon, save_con)
        
        print("Samples processed successfully.")

    def _read_cube(self, filepath):
        """Read data cube file."""
        cube_data = np.loadtxt(filepath)
        
        # Reshape to (ni, nj, nr) where nr is total wavelength bins
        expected_size = self.metadata.ni * self.metadata.nj * self.metadata.nr
        
        if cube_data.size != expected_size:
            raise ValueError(
                f"Data cube size mismatch: expected {expected_size} elements "
                f"({self.metadata.ni}×{self.metadata.nj}×{self.metadata.nr}), "
                f"got {cube_data.size}"
            )
        
        return cube_data.reshape(self.metadata.naxis)

    def _process_samples(self, samples, save_maps, save_precon, save_con):
        """Process samples based on what data was saved."""
        map_shp = self.metadata.naxis[:2].prod()
        
        # Process maps if saved
        if save_maps:
            self.maps = np.zeros((
                self.nsamples,
                self._nlines + 2,  # flux maps + velocity + velocity dispersion
                *self.metadata.naxis[:2]
            ))
            
            for s in range(self.nsamples):
                # Flux maps for each line
                for ln in range(self._nlines):
                    start_idx = ln * map_shp
                    end_idx = (ln + 1) * map_shp
                    self.maps[s, ln, :, :] = samples[s, start_idx:end_idx].reshape(
                        self.metadata.naxis[:2]
                    )

                # LoS velocity map
                start_idx = self._nlines * map_shp
                end_idx = (self._nlines + 1) * map_shp
                self.maps[s, self._nlines, :, :] = samples[s, start_idx:end_idx].reshape(
                    self.metadata.naxis[:2]
                )

                # LoS velocity dispersion map
                start_idx = (self._nlines + 1) * map_shp
                end_idx = (self._nlines + 2) * map_shp
                self.maps[s, self._nlines + 1, :, :] = samples[s, start_idx:end_idx].reshape(
                    self.metadata.naxis[:2]
                )

        # Track position in samples array
        sample_idx = save_maps * (2 + self._nlines) * map_shp

        # Process preconvolved cubes if saved
        if save_precon:
            self.precon_cubes = np.zeros((self.nsamples, *self.metadata.naxis))
            for s in range(self.nsamples):
                start_idx = sample_idx
                end_idx = sample_idx + self.metadata.sz
                self.precon_cubes[s, :, :, :] = samples[s, start_idx:end_idx].reshape(
                    self.metadata.naxis
                )
            sample_idx += self.metadata.sz

        # Process convolved cubes if saved
        if save_con:
            self.con_cubes = np.zeros((self.nsamples, *self.metadata.naxis))
            for s in range(self.nsamples):
                start_idx = sample_idx
                end_idx = sample_idx + self.metadata.sz
                self.con_cubes[s, :, :, :] = samples[s, start_idx:end_idx].reshape(
                    self.metadata.naxis
                )
            sample_idx += self.metadata.sz

        # Process blob parameters
        self._process_blob_parameters(samples, sample_idx)

    def _process_blob_parameters(self, samples, start_idx):
        """Process blob parameters from samples."""
        # Extract number of blobs
        self.max_blobs = int(samples[0, start_idx + 1])

        # Global parameter names
        global_names = ['WD', 'RADMIN', 'RADMAX', 'QMIN']
        for i in range(self._nlines):
            global_names += [f'FLUX{i}MU', f'FLUX{i}SD']
        global_names += [
            'NUMBLOBS', 'XC', 'YC', 'DISKFLUX', 'DISKMU',
            'VSYS', 'VMAX', 'VSLOPE', 'VGAMMA', 'VBETA'
        ]
        global_names += [f'VDISP{i}' for i in range(self._nsigmad)]
        global_names += ['INC', 'PA', 'SIGMA0', 'SIGMA1']

        # Extract global parameters
        global_start = start_idx + 2
        global_end = global_start + 5 + 2 * self._nlines
        global_param_data = np.concatenate((
            samples[:, global_start:global_end],
            samples[:, -13 - self._nsigmad:]
        ), axis=1)
        
        self.global_param = pd.DataFrame(global_param_data, columns=global_names)
        self.global_param.index.name = 'SAMPLE'

        # Extract blob parameters if any blobs exist
        if self.max_blobs > 0:
            blob_names = ['RC', 'THETAC', 'W', 'Q', 'PHI']
            blob_names += [f'FLUX{i}' for i in range(self._nlines)]
            n_bparam = len(blob_names)
            
            self.blob_param = np.zeros((self.nsamples * self.max_blobs, n_bparam))
            
            blob_start = global_end
            blob_end = -13 - self._nsigmad
            
            for s in range(self.nsamples):
                row_start = self.max_blobs * s
                row_end = self.max_blobs * (s + 1)
                
                sblob_param = samples[s, blob_start:blob_end]
                sblob_param = sblob_param.reshape((self.max_blobs, n_bparam), order='F')
                self.blob_param[row_start:row_end, :] = sblob_param

            self.blob_param = pd.DataFrame(self.blob_param, columns=blob_names)
            self.blob_param['SAMPLE'] = np.repeat(np.arange(self.nsamples), self.max_blobs)
            self.blob_param['BLOB'] = np.tile(np.arange(self.max_blobs), self.nsamples)
            self.blob_param.set_index(['SAMPLE', 'BLOB'], inplace=True)
            self.blob_param = self.blob_param[self.blob_param['RC'] > 0.0]

    def get_wavelength_slice(self, wavelength, tolerance=None):
        """Get a wavelength slice from cubes for a specific wavelength.
        
        Parameters
        ----------
        wavelength : float
            Target wavelength in Angstroms
        tolerance : float, optional
            Maximum allowed difference from target wavelength. If None,
            uses the wavelength resolution of the containing window.
            
        Returns
        -------
        dict
            Dictionary containing slices from available cubes and metadata
        """
        # Find the global index for this wavelength
        global_idx = self.metadata.get_global_wavelength_index(wavelength)
        
        if global_idx is None:
            raise ValueError(f"Wavelength {wavelength} Å not found in any window")
        
        # Get actual wavelength at this index
        actual_wavelength = self.metadata.r_full[global_idx]
        
        # Check tolerance if specified
        if tolerance is not None:
            if abs(actual_wavelength - wavelength) > tolerance:
                raise ValueError(
                    f"Closest wavelength {actual_wavelength} Å exceeds tolerance "
                    f"{tolerance} Å from target {wavelength} Å"
                )
        
        result = {
            'target_wavelength': wavelength,
            'actual_wavelength': actual_wavelength,
            'global_index': global_idx,
            'data_slice': self.data[:, :, global_idx],
            'var_slice': self.var[:, :, global_idx]
        }
        
        # Add model slices if available
        if hasattr(self, 'precon_cubes'):
            result['precon_slices'] = self.precon_cubes[:, :, :, global_idx]
        if hasattr(self, 'con_cubes'):
            result['con_slices'] = self.con_cubes[:, :, :, global_idx]
            
        return result

    def get_window_spectrum(self, i, j, window_idx=None):
        """Get spectrum for a specific spatial pixel, optionally from a specific window.
        
        Parameters
        ----------
        i, j : int
            Spatial pixel coordinates
        window_idx : int, optional
            If specified, return only spectrum from this window
            
        Returns
        -------
        dict
            Dictionary containing wavelength array and spectra
        """
        if window_idx is not None:
            if window_idx >= len(self.metadata.wavelength_windows):
                raise ValueError(f"Window index {window_idx} out of range")
            
            window = self.metadata.wavelength_windows[window_idx]
            wave_slice = slice(window.start_idx, window.end_idx + 1)
            
            result = {
                'wavelength': window.r,
                'data_spectrum': self.data[i, j, wave_slice],
                'var_spectrum': self.var[i, j, wave_slice],
                'window_info': {
                    'index': window_idx,
                    'r_min': window.r_min,
                    'r_max': window.r_max,
                    'n_bins': window.n_bins
                }
            }
            
            # Add model spectra if available
            if hasattr(self, 'precon_cubes'):
                result['precon_spectra'] = self.precon_cubes[:, i, j, wave_slice]
            if hasattr(self, 'con_cubes'):
                result['con_spectra'] = self.con_cubes[:, i, j, wave_slice]
                
        else:
            # Return full spectrum across all windows
            result = {
                'wavelength': self.metadata.r_full,
                'data_spectrum': self.data[i, j, :],
                'var_spectrum': self.var[i, j, :],
                'window_info': [
                    {
                        'index': idx,
                        'r_min': w.r_min,
                        'r_max': w.r_max,
                        'n_bins': w.n_bins,
                        'start_idx': w.start_idx,
                        'end_idx': w.end_idx
                    }
                    for idx, w in enumerate(self.metadata.wavelength_windows)
                ]
            }
            
            # Add model spectra if available
            if hasattr(self, 'precon_cubes'):
                result['precon_spectra'] = self.precon_cubes[:, i, j, :]
            if hasattr(self, 'con_cubes'):
                result['con_spectra'] = self.con_cubes[:, i, j, :]
        
        return result

    def integrated_flux_map(self, window_idx=None, sample_idx=0):
        """Calculate integrated flux map over wavelength range.
        
        Parameters
        ----------
        window_idx : int, optional
            If specified, integrate only over this window
        sample_idx : int, optional
            Sample index for model cubes (default: 0)
            
        Returns
        -------
        dict
            Dictionary containing integrated flux maps
        """
        if window_idx is not None:
            if window_idx >= len(self.metadata.wavelength_windows):
                raise ValueError(f"Window index {window_idx} out of range")
            
            window = self.metadata.wavelength_windows[window_idx]
            wave_slice = slice(window.start_idx, window.end_idx + 1)
            dr = window.dr
            
        else:
            # Integrate over all windows
            wave_slice = slice(None)
            # Use average dr for integration
            dr = self.metadata.dr
        
        result = {
            'data_flux': np.sum(self.data[:, :, wave_slice], axis=2) * dr,
            'window_idx': window_idx
        }
        
        # Add model flux maps if available
        if hasattr(self, 'precon_cubes'):
            result['precon_flux'] = np.sum(
                self.precon_cubes[sample_idx, :, :, wave_slice], axis=2
            ) * dr
        if hasattr(self, 'con_cubes'):
            result['con_flux'] = np.sum(
                self.con_cubes[sample_idx, :, :, wave_slice], axis=2
            ) * dr
            
        return result

    def print_window_summary(self):
        """Print summary of wavelength windows."""
        print(f"\nWavelength Window Summary:")
        print(f"Total windows: {len(self.metadata.wavelength_windows)}")
        print(f"Total wavelength bins: {self.metadata.nr}")
        print(f"Spatial dimensions: {self.metadata.ni} × {self.metadata.nj}")
        print()
        
        for i, window in enumerate(self.metadata.wavelength_windows):
            print(f"Window {i+1}:")
            print(f"  Range: {window.r_min:.3f} - {window.r_max:.3f} Å")
            print(f"  Bins: {window.n_bins} (global indices {window.start_idx}-{window.end_idx})")
            print(f"  Resolution: {window.dr:.4f} Å/bin")
            print(f"  Original bins: {window.orig_start_bin}-{window.orig_end_bin}")

    def validate_emission_lines(self, emission_lines):
        """Validate that emission lines are within wavelength windows.
        
        Parameters
        ----------
        emission_lines : list
            List of emission line wavelengths in Angstroms
            
        Returns
        -------
        validation_results : dict
            Dictionary with validation results for each line
        """
        results = {}
        
        for line_wave in emission_lines:
            window = self.metadata.get_window_for_wavelength(line_wave)
            results[line_wave] = {
                'found': window is not None,
                'window_idx': None if window is None else self.metadata.wavelength_windows.index(window),
                'window_range': None if window is None else (window.r_min, window.r_max)
            }
        
        return results

    def extract_line_maps(self, line_wavelength, sample_idx=0, integration_width=2.0):
        """Extract integrated line maps around a specific wavelength.
        
        Parameters
        ----------
        line_wavelength : float
            Central wavelength of the line in Angstroms
        sample_idx : int, optional
            Sample index for model cubes (default: 0)
        integration_width : float, optional
            Integration width around line center in Angstroms (default: 2.0)
            
        Returns
        -------
        line_maps : dict
            Dictionary containing integrated line maps
        """
        # Find wavelength indices for integration
        wave_array = self.metadata.r_full
        center_idx = self.metadata.get_global_wavelength_index(line_wavelength)
        
        if center_idx is None:
            raise ValueError(f"Line wavelength {line_wavelength} Å not found in data")
        
        # Find integration range
        half_width = integration_width / 2.0
        min_wave = line_wavelength - half_width
        max_wave = line_wavelength + half_width
        
        # Get indices within integration range
        wave_mask = (wave_array >= min_wave) & (wave_array <= max_wave)
        integration_indices = np.where(wave_mask)[0]
        
        if len(integration_indices) == 0:
            raise ValueError(f"No wavelength bins found within {integration_width} Å of {line_wavelength} Å")
        
        # Calculate average wavelength spacing for flux scaling
        if len(integration_indices) > 1:
            dr_avg = np.mean(np.diff(wave_array[integration_indices]))
        else:
            # Find the wavelength resolution from the containing window
            window = self.metadata.get_window_for_wavelength(line_wavelength)
            dr_avg = window.dr if window else 1.0
        
        # Integrate over wavelength range
        result = {
            'line_wavelength': line_wavelength,
            'integration_range': (min_wave, max_wave),
            'n_bins': len(integration_indices),
            'wavelength_indices': integration_indices,
            'actual_wavelengths': wave_array[integration_indices],
            'data_map': np.sum(self.data[:, :, integration_indices], axis=2) * dr_avg,
            'var_map': np.sum(self.var[:, :, integration_indices], axis=2) * dr_avg**2,
            'error_map': np.sqrt(np.sum(self.var[:, :, integration_indices], axis=2)) * dr_avg
        }
        
        # Add model maps if available
        if hasattr(self, 'precon_cubes'):
            result['precon_map'] = np.sum(
                self.precon_cubes[sample_idx, :, :, integration_indices], axis=2
            ) * dr_avg
            
        if hasattr(self, 'con_cubes'):
            result['con_map'] = np.sum(
                self.con_cubes[sample_idx, :, :, integration_indices], axis=2
            ) * dr_avg
            
        return result

    def compare_windows(self, i, j, normalize=True):
        """Compare spectra between different windows for a given pixel.
        
        Parameters
        ----------
        i, j : int
            Spatial pixel coordinates
        normalize : bool, optional
            Whether to normalize spectra for comparison (default: True)
            
        Returns
        -------
        comparison : dict
            Dictionary containing comparison data
        """
        if len(self.metadata.wavelength_windows) < 2:
            raise ValueError("Need at least 2 windows for comparison")
        
        comparison = {
            'pixel': (i, j),
            'windows': []
        }
        
        for window_idx, window in enumerate(self.metadata.wavelength_windows):
            wave_slice = slice(window.start_idx, window.end_idx + 1)
            
            data_spectrum = self.data[i, j, wave_slice]
            var_spectrum = self.var[i, j, wave_slice]
            
            if normalize:
                # Normalize by median flux in window
                median_flux = np.median(data_spectrum[data_spectrum > 0])
                if median_flux > 0:
                    data_spectrum = data_spectrum / median_flux
                    var_spectrum = var_spectrum / median_flux**2
            
            window_data = {
                'window_idx': window_idx,
                'wavelength': window.r.copy(),
                'data_spectrum': data_spectrum,
                'var_spectrum': var_spectrum,
                'error_spectrum': np.sqrt(var_spectrum),
                'window_info': {
                    'r_min': window.r_min,
                    'r_max': window.r_max,
                    'n_bins': window.n_bins,
                    'dr': window.dr
                }
            }
            
            # Add model spectra if available
            if hasattr(self, 'con_cubes'):
                model_spectrum = self.con_cubes[0, i, j, wave_slice]  # Use first sample
                if normalize and median_flux > 0:
                    model_spectrum = model_spectrum / median_flux
                window_data['model_spectrum'] = model_spectrum
            
            comparison['windows'].append(window_data)
        
        return comparison

    def get_continuum_estimate(self, i, j, method='median', exclude_lines=None):
        """Estimate continuum level for a given pixel.
        
        Parameters
        ----------
        i, j : int
            Spatial pixel coordinates
        method : str, optional
            Method for continuum estimation ('median', 'percentile', 'edges')
        exclude_lines : list, optional
            List of line wavelengths to exclude from continuum estimate
            
        Returns
        -------
        continuum_info : dict
            Dictionary containing continuum estimate information
        """
        full_spectrum = self.data[i, j, :]
        full_wave = self.metadata.r_full
        full_var = self.var[i, j, :]
        
        # Create mask for continuum regions
        continuum_mask = np.ones(len(full_spectrum), dtype=bool)
        
        # Exclude emission lines if specified
        if exclude_lines is not None:
            for line_wave in exclude_lines:
                # Exclude ±3 Å around each line
                line_mask = np.abs(full_wave - line_wave) > 3.0
                continuum_mask &= line_mask
        
        # Apply method-specific selection
        continuum_flux = full_spectrum[continuum_mask]
        continuum_wave = full_wave[continuum_mask]
        continuum_var = full_var[continuum_mask]
        
        if len(continuum_flux) == 0:
            return {'continuum_level': 0.0, 'method': method, 'n_points': 0}
        
        if method == 'median':
            continuum_level = np.median(continuum_flux)
        elif method == 'percentile':
            continuum_level = np.percentile(continuum_flux, 25)  # 25th percentile
        elif method == 'edges':
            # Use flux from window edges
            edge_indices = []
            for window in self.metadata.wavelength_windows:
                # First and last few bins of each window
                start_idx = window.start_idx
                end_idx = window.end_idx
                n_edge = min(3, window.n_bins // 4)  # Use up to 3 bins or 1/4 of window
                
                edge_indices.extend(range(start_idx, start_idx + n_edge))
                edge_indices.extend(range(end_idx - n_edge + 1, end_idx + 1))
            
            edge_flux = full_spectrum[edge_indices]
            continuum_level = np.median(edge_flux) if len(edge_flux) > 0 else 0.0
        else:
            raise ValueError(f"Unknown continuum method: {method}")
        
        return {
            'continuum_level': continuum_level,
            'method': method,
            'n_points': len(continuum_flux),
            'continuum_wavelengths': continuum_wave,
            'continuum_flux': continuum_flux,
            'continuum_var': continuum_var
        }

    # Legacy plotting methods (commented out but structure preserved for reference)
    # These would need matplotlib imports and the b3dplot/b3dcomp modules
    
    # def plot_global_marginalised(self, save_file=None):
    #     """Plot marginalised distributions of global parameters."""
    #     pass
    
    # def plot_map(self, ax, map_2d, **kwargs):
    #     """Plot individual map to a given axes object."""
    #     pass
    
    # def setup_comparison_maps(self, figsize=(10.0, 10.0), log_flux=True, **kwargs):
    #     """Setup comparison maps for a given sample."""
    #     pass