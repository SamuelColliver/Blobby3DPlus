"""Enhanced SpectralModel with better line definition support.

This version improves the line parsing to handle the exact format from MODEL_OPTIONS
and provides better multi-window LSF support.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

from .const import PhysicalConstants


class SpectralModel:

    def __init__(self, lines=None, lsf_fwhm=None, baseline_order=None, wave_ref=0.0, 
                 wavelength_windows=None):
        """
        Parameters
        ----------
        lines : list of lists
            Emission line definitions in MODEL_OPTIONS format:
            - Single line: [wavelength]
            - Coupled lines: [main_wavelength, coupled_wavelength1, ratio1, coupled_wavelength2, ratio2, ...]
        lsf_fwhm : float or dict, optional
            Full-Width Half Maximum of Line Spread Function. Can be:
            - float: single LSF FWHM for all wavelengths
            - dict: mapping of window_idx to LSF FWHM values
            - None: will be extracted from wavelength_windows if available
        baseline_order : int, optional
            Order of baseline polynomial. The default is None.
        wave_ref : float, optional
            Redshift offset in Angstroms. The default is 0.0.
        wavelength_windows : list of WavelengthWindow, optional
            Wavelength windows for multi-window data. If None, assumes
            single continuous wavelength range.

        Returns
        -------
        None.
        """
        # process and validate emission lines
        self.lines = self._process_emission_lines(lines)
        self.nlines = len(self.lines)
        self.wavelength_windows = wavelength_windows

        # handle LSF FWHM - support both single value and per-window values
        if wavelength_windows is not None:
            if lsf_fwhm is None:
                # extract LSF FWHM from wavelength windows
                self.lsf_fwhm_map = {}
                self.lsf_sigma_map = {}
                for i, window in enumerate(wavelength_windows):
                    if hasattr(window, 'fwhm_lsf'):
                        self.lsf_fwhm_map[i] = window.fwhm_lsf
                        self.lsf_sigma_map[i] = window.fwhm_lsf / np.sqrt(8.0 * np.log(2.0))
                    else:
                        # fallback for windows without LSF info
                        self.lsf_fwhm_map[i] = 2.0  # default value
                        self.lsf_sigma_map[i] = 2.0 / np.sqrt(8.0 * np.log(2.0))
            elif isinstance(lsf_fwhm, dict):
                # user-provided per-window LSF values
                self.lsf_fwhm_map = lsf_fwhm.copy()
                self.lsf_sigma_map = {k: v/np.sqrt(8.0*np.log(2.0)) for k, v in lsf_fwhm.items()}
            else:
                # single LSF value for all windows
                self.lsf_fwhm_map = {i: lsf_fwhm for i in range(len(wavelength_windows))}
                self.lsf_sigma_map = {i: lsf_fwhm/np.sqrt(8.0*np.log(2.0)) for i in range(len(wavelength_windows))}
            
            # for backwards compatibility
            self.lsf_fwhm = lsf_fwhm if isinstance(lsf_fwhm, (int, float)) else list(self.lsf_fwhm_map.values())[0]
            self.lsf_sigma = self.lsf_fwhm / np.sqrt(8.0*np.log(2.0))
        else:
            # single window case
            self.lsf_fwhm = lsf_fwhm if lsf_fwhm is not None else 2.0
            self.lsf_sigma = self.lsf_fwhm / np.sqrt(8.0*np.log(2.0))
            self.lsf_fwhm_map = {0: self.lsf_fwhm}
            self.lsf_sigma_map = {0: self.lsf_sigma}

        self.baseline_order = baseline_order
        self.wave_ref = wave_ref

        self.nparam = self.nlines + 2
        if self.baseline_order is not None:
            self.nparam += 1 + self.baseline_order

    def _process_emission_lines(self, lines):
        """
        Process emission line definitions from MODEL_OPTIONS format.
        
        Parameters
        ----------
        lines : list of lists
            Raw line definitions from MODEL_OPTIONS parsing
            
        Returns
        -------
        processed_lines : list of lists
            Processed line definitions in internal format
        """
        if lines is None:
            # default to H-alpha if no lines specified
            return [[6562.81]]
        
        processed_lines = []
        
        for line_def in lines:
            if not line_def:
                continue
                
            # ensure we have at least one wavelength
            if len(line_def) < 1:
                continue
            
            # convert to internal format: [main_wavelength, coupled_wavelength1, ratio1, ...]
            processed_line = [float(line_def[0])]  # main wavelength
            
            # process coupled lines (pairs of wavelength and ratio)
            if len(line_def) > 1:
                # check if we have pairs of coupled wavelength and ratio
                if (len(line_def) - 1) % 2 != 0:
                    print(f"Warning: Line definition {line_def} has unpaired coupled line data. "
                          f"Expected format: [main_wave, coupled_wave1, ratio1, coupled_wave2, ratio2, ...]")
                    # truncate to complete pairs
                    n_pairs = (len(line_def) - 1) // 2
                    line_def = line_def[:1 + 2*n_pairs]
                
                # add coupled lines
                for i in range(1, len(line_def), 2):
                    if i + 1 < len(line_def):
                        coupled_wavelength = float(line_def[i])
                        ratio = float(line_def[i + 1])
                        processed_line.extend([coupled_wavelength, ratio])
            
            processed_lines.append(processed_line)
        
        return processed_lines

    def get_line_info(self):
        """
        Get human-readable information about the emission lines.
        
        Returns
        -------
        line_info : list of dict
            Information about each line group
        """
        line_info = []
        
        for i, line in enumerate(self.lines):
            info = {
                'line_group': i + 1,
                'main_wavelength': line[0],
                'coupled_lines': []
            }
            
            # process coupled lines
            if len(line) > 1:
                for j in range(1, len(line), 2):
                    if j + 1 < len(line):
                        coupled_info = {
                            'wavelength': line[j],
                            'ratio': line[j + 1]
                        }
                        info['coupled_lines'].append(coupled_info)
            
            line_info.append(info)
        
        return line_info

    def get_lsf_sigma_for_wavelength(self, wavelength):
        """Get LSF sigma for a specific wavelength."""
        if self.wavelength_windows is None:
            return self.lsf_sigma
        
        # find which window contains this wavelength
        for i, window in enumerate(self.wavelength_windows):
            if window.r_min <= wavelength <= window.r_max:
                return self.lsf_sigma_map.get(i, self.lsf_sigma)
        
        # fallback to default if wavelength not in any window
        return self.lsf_sigma

    def validate_lines_in_windows(self):
        """
        Validate that all emission lines fall within the defined wavelength windows.
        
        Returns
        -------
        validation_results : dict
            Results of validation for each line
        """
        if self.wavelength_windows is None:
            return {'status': 'no_windows', 'message': 'No wavelength windows defined'}
        
        results = {
            'all_valid': True,
            'line_results': [],
            'warnings': []
        }
        
        for i, line in enumerate(self.lines):
            line_result = {
                'line_group': i + 1,
                'main_wavelength': line[0],
                'main_valid': False,
                'main_window': None,
                'coupled_results': []
            }
            
            # check main wavelength
            for w_idx, window in enumerate(self.wavelength_windows):
                if window.r_min <= line[0] <= window.r_max:
                    line_result['main_valid'] = True
                    line_result['main_window'] = w_idx
                    break
            
            if not line_result['main_valid']:
                results['all_valid'] = False
                results['warnings'].append(
                    f"Main wavelength {line[0]} Å in line group {i+1} not found in any window"
                )
            
            # check coupled lines
            if len(line) > 1:
                for j in range(1, len(line), 2):
                    if j + 1 < len(line):
                        coupled_wave = line[j]
                        ratio = line[j + 1]
                        
                        coupled_result = {
                            'wavelength': coupled_wave,
                            'ratio': ratio,
                            'valid': False,
                            'window': None
                        }
                        
                        for w_idx, window in enumerate(self.wavelength_windows):
                            if window.r_min <= coupled_wave <= window.r_max:
                                coupled_result['valid'] = True
                                coupled_result['window'] = w_idx
                                break
                        
                        if not coupled_result['valid']:
                            results['all_valid'] = False
                            results['warnings'].append(
                                f"Coupled wavelength {coupled_wave} Å in line group {i+1} not found in any window"
                            )
                        
                        line_result['coupled_results'].append(coupled_result)
            
            results['line_results'].append(line_result)
        
        return results

    def print_line_summary(self):
        """Print a summary of the emission line configuration."""
        print(f"\nSpectralModel Line Configuration:")
        print(f"Number of line groups: {self.nlines}")
        
        line_info = self.get_line_info()
        for info in line_info:
            print(f"\nLine Group {info['line_group']}:")
            print(f"  Main wavelength: {info['main_wavelength']:.2f} Å")
            
            if info['coupled_lines']:
                print(f"  Coupled lines:")
                for coupled in info['coupled_lines']:
                    print(f"    {coupled['wavelength']:.2f} Å (ratio: {coupled['ratio']:.4f})")
            else:
                print(f"  No coupled lines")
        
        # print LSF information
        if hasattr(self, 'lsf_fwhm_map') and len(self.lsf_fwhm_map) > 1:
            print(f"\nLSF FWHM per window:")
            for window_idx, fwhm in self.lsf_fwhm_map.items():
                print(f"  Window {window_idx}: {fwhm:.3f} Å")
        else:
            print(f"\nGlobal LSF FWHM: {self.lsf_fwhm:.3f} Å")
        
        # validate lines if windows are defined
        if self.wavelength_windows is not None:
            validation = self.validate_lines_in_windows()
            if validation['all_valid']:
                print(f"\n✓ All emission lines are within the wavelength windows")
            else:
                print(f"\n⚠ Some emission lines are outside the wavelength windows:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")

    # rest of the methods remain the same as in the original implementation
    def calculate(self, wave, *param):
        """Calculate spectral model."""
        if self.baseline_order is None:
            model = self._gas_model(wave, param)
        else:
            model = self._gas_model(wave, param[:self.nlines+2])
            model += self._baseline_model(wave, param[-1-self.baseline_order:])

        return model

    def calculate_cube(self, wave, param):
        """Calculate spectral model for cube."""
        model_cube = np.zeros((*param.shape[1:], len(wave)))*np.nan
        for i in range(param.shape[1]):
            for j in range(param.shape[2]):
                model_cube[i, j] = self.calculate(wave, *param[:, i, j])

        return model_cube

    def fit_spaxel(self, wave, data, var=None, bounds=None):
        """Fit spaxel using curve_fit."""
        if bounds is None:
            # line flux bounds
            bounds = [
                [0.0]*self.nlines,
                [np.inf]*self.nlines,
            ]

            # v, vdisp bounds
            bounds[0] += [-np.inf, 0.0]
            bounds[1] += [np.inf, np.inf]

            if self.baseline_order is not None:
                bounds[0] += [-np.inf]*(self.baseline_order + 1)
                bounds[1] += [np.inf]*(self.baseline_order + 1)

        if (var is None) & np.any(data != 0.0):
            data_valid = np.isfinite(data)
            data_tmp = data[data_valid]
            w_tmp = wave[data_valid]
            sigma_tmp = None
        elif (var is None) & np.all(data == 0.0):
            popt = np.zeros(self.nparam)*np.nan
            pcov = np.zeros(self.nparam)*np.nan
            return popt, pcov
        elif np.any(var > 0.0):
            data_valid = ((var > 0.0) & np.isfinite(var) & np.isfinite(data))
            data_tmp = data[data_valid]
            w_tmp = wave[data_valid]
            sigma_tmp = np.sqrt(var[data_valid])
        else:
            popt = np.zeros(self.nparam)*np.nan
            pcov = np.zeros(self.nparam)*np.nan
            return popt, pcov

        if len(w_tmp) <= 1:
            popt = np.zeros(self.nparam)*np.nan
            pcov = np.zeros(self.nparam)*np.nan
            return popt, pcov

        guess = self._guess(w_tmp, data_tmp)

        # enforce guess within bounds
        for i in range(len(guess)):
            if guess[i] < bounds[0][i]:
                guess[i] = bounds[0][i]
            elif guess[i] > bounds[1][i]:
                guess[i] = bounds[1][i]
            elif ~np.isfinite(guess[i]):
                guess[i] = 0.5*(bounds[1][i] - bounds[0][i])

        try:
            popt, pcov = curve_fit(
                self.calculate,
                w_tmp,
                data_tmp,
                sigma=sigma_tmp,
                bounds=bounds,
                p0=guess,
            )

            pcov = pcov.diagonal().copy()

        except RuntimeError:
            # occurs when curve_fit fails to converge
            popt = np.zeros(guess.size)*np.nan
            pcov = np.zeros(guess.size)*np.nan

        return popt, pcov

    def fit_cube(self, wave, data, var=None, bounds=None, wave_axis=2):
        """Fit cube using curve_fit."""
        if wave_axis == 2:
            shp = (self.nparam, *data.shape[:2])
            fit = np.zeros(shp)*np.nan
            fit_err = np.zeros(shp)*np.nan
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if var is None:
                        var_tmp = None
                    else:
                        var_tmp = var[i, j]

                    fit[:, i, j], fit_err[:, i, j] = self.fit_spaxel(
                        wave, data[i, j], var_tmp, bounds)

        elif wave_axis == 0:
            shp = (self.nparam, *data.shape[1:])
            fit = np.zeros(shp)*np.nan
            fit_err = np.zeros(shp)*np.nan
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    if var is None:
                        var_tmp = None
                    else:
                        var_tmp = var[:, i, j]

                    fit[:, i, j], fit_err[:, i, j] = self.fit_spaxel(
                        wave, data[:, i, j], var_tmp, bounds)

        else:
            raise ValueError('Wave axis needs to be 0 or 2.')

        return fit, fit_err

    def fit_window_spectrum(self, wave, data, var=None, bounds=None, window_idx=None):
        """Fit spectrum from a specific wavelength window."""
        if self.wavelength_windows is None or window_idx is None:
            # fit full spectrum
            return self.fit_spaxel(wave, data, var, bounds)
        
        if window_idx >= len(self.wavelength_windows):
            raise ValueError(f"Window index {window_idx} out of range")
        
        # extract window data
        window = self.wavelength_windows[window_idx]
        wave_slice = slice(window.start_idx, window.end_idx + 1)
        
        wave_window = wave[wave_slice]
        data_window = data[wave_slice]
        var_window = var[wave_slice] if var is not None else None
        
        return self.fit_spaxel(wave_window, data_window, var_window, bounds)

    def _guess(self, wave, data, lambda_win=10.0):
        """Guess parameters using method of moments."""
        if len(wave) <= 1 or len(data) <= 1:
            # handle edge case of insufficient data
            if self.baseline_order is None:
                guess = np.zeros(self.nlines + 2)
            else:
                guess = np.zeros(self.nlines + 2 + self.baseline_order + 1)
            guess[self.nlines+1] = 50.0  # default velocity dispersion
            return guess
        
        dwave = np.median(np.diff(wave)) if len(wave) > 1 else 1.0
        wave_left = wave - 0.5*dwave
        wave_right = wave + 0.5*dwave

        if self.baseline_order is None:
            guess = np.zeros(self.nlines + 2)
        else:
            guess = np.zeros(self.nlines + 2 + self.baseline_order + 1)

        tmp_v = np.zeros(self.nlines)*np.nan
        tmp_vdisp = np.ones(self.nlines)*np.nan

        # check if data has any useful signal
        data_max = np.nanmax(data)
        data_min = np.nanmin(data)
        if data_max <= data_min or np.all(np.isnan(data)) or np.all(data <= 0):
            # no useful signal - return defaults
            guess[self.nlines] = 0.0  # velocity
            guess[self.nlines+1] = 50.0  # velocity dispersion
            return guess

        for i, line in enumerate(self.lines):
            # check if line is within the wavelength range (with some tolerance)
            if wave.min() > line[0] + lambda_win or wave.max() < line[0] - lambda_win:
                # line is completely outside wavelength range
                continue
                
            win = (
                (wave_right >= line[0] - lambda_win)
                & (wave_left <= line[0] + lambda_win)
            )
            
            if win.sum() <= 0:
                # if no valid pixels around emission line, use initial guesses
                continue

            win_data = data[win]
            win_wave = wave[win]

            # skip if all data in window is NaN, zero, or negative
            finite_mask = np.isfinite(win_data)
            if not np.any(finite_mask):
                continue
                
            win_data_clean = win_data[finite_mask]
            win_wave_clean = win_wave[finite_mask]
            
            if len(win_data_clean) == 0 or np.all(win_data_clean <= 0):
                continue

            # guess flux (use only positive values)
            positive_data = win_data_clean[win_data_clean > 0]
            if len(positive_data) > 0:
                guess[i] = max(0.0, positive_data.sum() * dwave)
            else:
                continue

            # calculate weights for velocity - normalise to prevent zero weights
            data_range = np.max(win_data_clean) - np.min(win_data_clean)
            if data_range > 0:
                weights_v = win_data_clean - np.min(win_data_clean)
                weights_v = np.maximum(weights_v, 1e-10)  # prevent exactly zero weights
            else:
                weights_v = np.ones(len(win_data_clean))

            if np.sum(weights_v) > 0:
                try:
                    mean_wave = np.average(win_wave_clean, weights=weights_v)
                    tmp_v[i] = (mean_wave/line[0] - 1.0)*PhysicalConstants.C
                except (ZeroDivisionError, ValueError):
                    tmp_v[i] = 0.0
            else:
                tmp_v[i] = 0.0

            # calculate velocity dispersion guess - use window-specific LSF
            lsf_sigma_for_line = self.get_lsf_sigma_for_wavelength(line[0])
            
            # use only positive data for velocity dispersion weights
            weights_vdisp = np.maximum(win_data_clean, 0.0)
            
            if np.sum(weights_vdisp) > 0 and not np.isnan(mean_wave):
                try:
                    var_wave = np.average((win_wave_clean - mean_wave)**2, weights=weights_vdisp)
                    var_wave = max(0, var_wave - lsf_sigma_for_line**2)
                    tmp_vdisp[i] = max(1.0, np.sqrt(var_wave) * PhysicalConstants.C / line[0])
                except (ZeroDivisionError, ValueError, RuntimeWarning):
                    tmp_vdisp[i] = 50.0  # default 50 km/s
            else:
                tmp_vdisp[i] = 50.0  # default 50 km/s

        # use weighted averages for velocity and dispersion with better error handling
        valid_flux = (guess[:self.nlines] > 0) & np.isfinite(guess[:self.nlines])
        valid_v = np.isfinite(tmp_v)
        valid_vdisp = np.isfinite(tmp_vdisp) & (tmp_vdisp > 0)
        
        # velocity estimate
        if np.any(valid_flux & valid_v):
            weights = guess[:self.nlines]
            mask = valid_flux & valid_v
            weights_masked = weights[mask]
            values_masked = tmp_v[mask]
            
            if np.sum(weights_masked) > 0:
                try:
                    guess[self.nlines] = np.average(values_masked, weights=weights_masked)
                except (ZeroDivisionError, ValueError):
                    guess[self.nlines] = 0.0
            else:
                guess[self.nlines] = 0.0
        else:
            guess[self.nlines] = 0.0
            
        # velocity dispersion estimate
        if np.any(valid_flux & valid_vdisp):
            weights = guess[:self.nlines]
            mask = valid_flux & valid_vdisp
            weights_masked = weights[mask]
            values_masked = tmp_vdisp[mask]
            
            if np.sum(weights_masked) > 0:
                try:
                    guess[self.nlines+1] = np.average(values_masked, weights=weights_masked)
                except (ZeroDivisionError, ValueError):
                    guess[self.nlines+1] = 50.0
            else:
                guess[self.nlines+1] = 50.0
        else:
            guess[self.nlines+1] = 50.0

        # handle any remaining NaN values and ensure reasonable bounds
        guess = np.nan_to_num(guess, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # ensure velocity dispersion is positive and reasonable (1-500 km/s)
        guess[self.nlines+1] = max(1.0, min(500.0, guess[self.nlines+1]))
        
        # ensure velocity is reasonable (-1000 to 1000 km/s)
        guess[self.nlines] = max(-1000.0, min(1000.0, guess[self.nlines]))

        return guess

    def _gas_model(self, wave, gas_param):
        """Generate gas emission model."""
        model = np.zeros(len(wave))

        rel_lambda = 1.0 + gas_param[self.nlines]/PhysicalConstants.C
        rel_lambda_sigma = gas_param[self.nlines+1]/PhysicalConstants.C

        # add emission line contribution
        for i, line in enumerate(self.lines):
            # model first line
            line_wave = line[0]
            line_flux = gas_param[i]

            lam = rel_lambda*line_wave
            lam_sigma = rel_lambda_sigma*line_wave

            model += self._gas_line_model(wave, line_flux, lam, lam_sigma, line_wave)

            # add coupled lines
            nclines = (len(line) - 1) // 2
            for j in range(nclines):
                coupled_idx = 1 + 2*j
                line_wave = line[coupled_idx]
                factor = line[coupled_idx + 1]
                lam = rel_lambda*line_wave
                lam_sigma = rel_lambda_sigma*line_wave
                model += self._gas_line_model(
                    wave, factor*line_flux, lam, lam_sigma, line_wave)

        return model

    def _gas_line_model(self, wave, flux, lam, lam_sigma, line_wave_rest):
        """Gas emission line model."""
        # get window-specific LSF sigma for this line
        lsf_sigma = self.get_lsf_sigma_for_wavelength(line_wave_rest)
        
        # handle potential irregular wavelength spacing
        if len(wave) == 1:
            dwave = 1.0  # single wavelength point
        else:
            # calculate local wavelength spacing
            dwave = np.zeros(len(wave))
            
            # first bin
            if len(wave) > 1:
                dwave[0] = wave[1] - wave[0]
            else:
                dwave[0] = 1.0
                
            # middle bins
            for i in range(1, len(wave) - 1):
                dwave[i] = 0.5 * (wave[i+1] - wave[i-1])
                
            # last bin
            if len(wave) > 1:
                dwave[-1] = wave[-1] - wave[-2]
            else:
                dwave[-1] = 1.0

        wave_left = wave - 0.5*dwave
        wave_right = wave + 0.5*dwave

        var = lam_sigma**2 + lsf_sigma**2

        cdf_left = 0.5*erf((wave_left - lam)/np.sqrt(2.0*var))
        cdf_right = 0.5*erf((wave_right - lam)/np.sqrt(2.0*var))

        return flux*(cdf_right - cdf_left)

    def _baseline_model(self, wave, baseline_param):
        """Baseline model."""
        wave_shft = wave - self.wave_ref
        baseline = np.ones(len(wave))*baseline_param[0]
        for i in range(self.baseline_order):
            baseline += baseline_param[1+i]*wave_shft**(1+i)

        return baseline