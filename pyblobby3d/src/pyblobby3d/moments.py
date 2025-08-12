"""Fitting kinematic moments with multi-window support.

@original author: Mathew Varidel
@edited by: Samuel Colliver
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

from .const import PhysicalConstants


class SpectralModel:

    def __init__(self, lines, lsf_fwhm=None, baseline_order=None, wave_ref=0.0, 
                 wavelength_windows=None):
        """
        Parameters
        ----------
        lines : list of lists
            Emission line definitions.
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
        self.lines = lines
        self.nlines = len(lines)
        self.wavelength_windows = wavelength_windows

        # Handle LSF FWHM - support both single value and per-window values
        if wavelength_windows is not None:
            if lsf_fwhm is None:
                # Extract LSF FWHM from wavelength windows
                self.lsf_fwhm_map = {}
                self.lsf_sigma_map = {}
                for i, window in enumerate(wavelength_windows):
                    if hasattr(window, 'fwhm_lsf'):
                        self.lsf_fwhm_map[i] = window.fwhm_lsf
                        self.lsf_sigma_map[i] = window.fwhm_lsf / np.sqrt(8.0 * np.log(2.0))
                    else:
                        # Fallback for windows without LSF info
                        self.lsf_fwhm_map[i] = 2.0  # Default value
                        self.lsf_sigma_map[i] = 2.0 / np.sqrt(8.0 * np.log(2.0))
            elif isinstance(lsf_fwhm, dict):
                # User-provided per-window LSF values
                self.lsf_fwhm_map = lsf_fwhm.copy()
                self.lsf_sigma_map = {k: v/np.sqrt(8.0*np.log(2.0)) for k, v in lsf_fwhm.items()}
            else:
                # Single LSF value for all windows
                self.lsf_fwhm_map = {i: lsf_fwhm for i in range(len(wavelength_windows))}
                self.lsf_sigma_map = {i: lsf_fwhm/np.sqrt(8.0*np.log(2.0)) for i in range(len(wavelength_windows))}
            
            # For backwards compatibility
            self.lsf_fwhm = lsf_fwhm if isinstance(lsf_fwhm, (int, float)) else list(self.lsf_fwhm_map.values())[0]
            self.lsf_sigma = self.lsf_fwhm / np.sqrt(8.0*np.log(2.0))
        else:
            # Single window case
            self.lsf_fwhm = lsf_fwhm if lsf_fwhm is not None else 2.0
            self.lsf_sigma = self.lsf_fwhm / np.sqrt(8.0*np.log(2.0))
            self.lsf_fwhm_map = {0: self.lsf_fwhm}
            self.lsf_sigma_map = {0: self.lsf_sigma}

        self.baseline_order = baseline_order
        self.wave_ref = wave_ref

        self.nparam = self.nlines + 2
        if self.baseline_order is not None:
            self.nparam += 1 + self.baseline_order

    def get_lsf_sigma_for_wavelength(self, wavelength):
        """Get LSF sigma for a specific wavelength."""
        if self.wavelength_windows is None:
            return self.lsf_sigma
        
        # Find which window contains this wavelength
        for i, window in enumerate(self.wavelength_windows):
            if window.r_min <= wavelength <= window.r_max:
                return self.lsf_sigma_map.get(i, self.lsf_sigma)
        
        # Fallback to default if wavelength not in any window
        return self.lsf_sigma

    def calculate(self, wave, *param):
        """Calculate spectral model.

        Parameters
        ----------
        wave : array-like
            Wavelength array.
        param : array-like
            Model parameters. The format is
            [flux_1, flux_2, .., velocity, velocity dispersion, b0, b1, ...].
            Where flux_i is the flux for the i-th coupled emission line and
            b0, b1, ... are the baseline polynomial coefficients.

        Returns
        -------
        model : np.ndarray
        """
        if self.baseline_order is None:
            model = self._gas_model(wave, param)
        else:
            model = self._gas_model(wave, param[:self.nlines+2])
            model += self._baseline_model(wave, param[-1-self.baseline_order:])

        return model

    def calculate_cube(self, wave, param):
        """Calculate spectral model for cube.

        Parameters
        ----------
        wave : array-like
            Wavelength array.
        param : 3D np.ndarray
            Cube of model parameters. For each spaxel the parameter format is
            [flux_1, flux_2, .., velocity, velocity dispersion]. Where flux_i
            is the flux for the i-th coupled emission line.

        Returns
        -------
        model : np.ndarray
            Model cube with axis (i, j, wavelength).
        """
        model_cube = np.zeros((*param.shape[1:], len(wave)))*np.nan
        for i in range(param.shape[1]):
            for j in range(param.shape[2]):
                model_cube[i, j] = self.calculate(wave, *param[:, i, j])

        return model_cube

    def fit_spaxel(self, wave, data, var=None, bounds=None):
        """Fit spaxel using curve_fit.

        Parameters
        ----------
        wave : array-like
            Wavelength array.
        data : array-like
        var : array-like, default is None
            Variance array. Default is None, which means curve_fit is not
            supplied a variance array.
        bounds : list of lists
            Bounds supplied to curve_fit for each parameter.

        Returns
        -------
        fit : np.ndarray
            Fit parameters returned by curve_fit.
        fit_err : np.ndarray
            Fit errors returned by curve_fit.
        """
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
            # Occurs when curve_fit fails to converge
            popt = np.zeros(guess.size)*np.nan
            pcov = np.zeros(guess.size)*np.nan

        return popt, pcov

    def fit_cube(self, wave, data, var=None, bounds=None, wave_axis=2):
        """Fit cube using curve_fit.

        Parameters
        ----------
        wave : array-like
            Wavelength array.
        data : array-like
        var : array-like, default is None
            Variance array. Default is None, which means curve_fit is not
            supplied a variance array.
        bounds : list of lists
            Bounds supplied to curve_fit for each parameter.
        wave_axis : int
            Axis for wavelength can either be 0 or 2. Default is 2.

        Returns
        -------
        fit : np.ndarray
            Fit parameters returned by curve_fit.
        fit_err : np.ndarray
            Fit errors returned by curve_fit.
        """
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
        """Fit spectrum from a specific wavelength window.
        
        Parameters
        ----------
        wave : array-like
            Full wavelength array
        data : array-like
            Full spectrum data
        var : array-like, optional
            Full spectrum variance
        bounds : list, optional
            Parameter bounds
        window_idx : int, optional
            Index of wavelength window to fit. If None, fits all windows.
            
        Returns
        -------
        fit : np.ndarray
            Fit parameters
        fit_err : np.ndarray
            Fit errors
        """
        if self.wavelength_windows is None or window_idx is None:
            # Fit full spectrum
            return self.fit_spaxel(wave, data, var, bounds)
        
        if window_idx >= len(self.wavelength_windows):
            raise ValueError(f"Window index {window_idx} out of range")
        
        # Extract window data
        window = self.wavelength_windows[window_idx]
        wave_slice = slice(window.start_idx, window.end_idx + 1)
        
        wave_window = wave[wave_slice]
        data_window = data[wave_slice]
        var_window = var[wave_slice] if var is not None else None
        
        return self.fit_spaxel(wave_window, data_window, var_window, bounds)

    def _guess(self, wave, data, lambda_win=10.0):
        """Guess parameters using method of moments.

        This takes a window around each emission line then calculates the
        mean and standard deviation. For coupled lines it just uses the
        first line in the lines array to make a guess.

        Parameters
        ----------
        wave : array-like
            Wavelength array.
        data : array-like
        lambda_win : float
            Window to estimate parameters

        Returns
        -------
        guess : np.ndarray
        """
        if len(wave) <= 1 or len(data) <= 1:
            # Handle edge case of insufficient data
            if self.baseline_order is None:
                guess = np.zeros(self.nlines + 2)
            else:
                guess = np.zeros(self.nlines + 2 + self.baseline_order + 1)
            guess[self.nlines+1] = 50.0  # Default velocity dispersion
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

        # Check if data has any useful signal
        data_max = np.nanmax(data)
        data_min = np.nanmin(data)
        if data_max <= data_min or np.all(np.isnan(data)) or np.all(data <= 0):
            # No useful signal - return defaults
            guess[self.nlines] = 0.0  # velocity
            guess[self.nlines+1] = 50.0  # velocity dispersion
            return guess

        for i, line in enumerate(self.lines):
            # Check if line is within the wavelength range (with some tolerance)
            if wave.min() > line[0] + lambda_win or wave.max() < line[0] - lambda_win:
                # Line is completely outside wavelength range
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

            # Skip if all data in window is NaN, zero, or negative
            finite_mask = np.isfinite(win_data)
            if not np.any(finite_mask):
                continue
                
            win_data_clean = win_data[finite_mask]
            win_wave_clean = win_wave[finite_mask]
            
            if len(win_data_clean) == 0 or np.all(win_data_clean <= 0):
                continue

            # Guess flux (use only positive values)
            positive_data = win_data_clean[win_data_clean > 0]
            if len(positive_data) > 0:
                guess[i] = max(0.0, positive_data.sum() * dwave)
            else:
                continue

            # Calculate weights for velocity - normalize to prevent zero weights
            data_range = np.max(win_data_clean) - np.min(win_data_clean)
            if data_range > 0:
                weights_v = win_data_clean - np.min(win_data_clean)
                weights_v = np.maximum(weights_v, 1e-10)  # Prevent exactly zero weights
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

            # Calculate velocity dispersion guess - use window-specific LSF
            lsf_sigma_for_line = self.get_lsf_sigma_for_wavelength(line[0])
            
            # Use only positive data for velocity dispersion weights
            weights_vdisp = np.maximum(win_data_clean, 0.0)
            
            if np.sum(weights_vdisp) > 0 and not np.isnan(mean_wave):
                try:
                    var_wave = np.average((win_wave_clean - mean_wave)**2, weights=weights_vdisp)
                    var_wave = max(0, var_wave - lsf_sigma_for_line**2)
                    tmp_vdisp[i] = max(1.0, np.sqrt(var_wave) * PhysicalConstants.C / line[0])
                except (ZeroDivisionError, ValueError, RuntimeWarning):
                    tmp_vdisp[i] = 50.0  # Default 50 km/s
            else:
                tmp_vdisp[i] = 50.0  # Default 50 km/s

        # Use weighted averages for velocity and dispersion with better error handling
        valid_flux = (guess[:self.nlines] > 0) & np.isfinite(guess[:self.nlines])
        valid_v = np.isfinite(tmp_v)
        valid_vdisp = np.isfinite(tmp_vdisp) & (tmp_vdisp > 0)
        
        # Velocity estimate
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
            
        # Velocity dispersion estimate
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

        # Handle any remaining NaN values and ensure reasonable bounds
        guess = np.nan_to_num(guess, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Ensure velocity dispersion is positive and reasonable (1-500 km/s)
        guess[self.nlines+1] = max(1.0, min(500.0, guess[self.nlines+1]))
        
        # Ensure velocity is reasonable (-1000 to 1000 km/s)
        guess[self.nlines] = max(-1000.0, min(1000.0, guess[self.nlines]))

        return guess

    def _gas_model(self, wave, gas_param):
        """
        Parameters
        ----------
        wave : array-like
            Wavelength array.
        gas_param : array-like
            Gas parameters.

        Returns
        -------
        model : np.ndarray
        """
        model = np.zeros(len(wave))

        rel_lambda = 1.0 + gas_param[self.nlines]/PhysicalConstants.C
        rel_lambda_sigma = gas_param[self.nlines+1]/PhysicalConstants.C

        # Add emission line contribution
        for i, line in enumerate(self.lines):
            # Model first line
            line_wave = line[0]
            line_flux = gas_param[i]

            lam = rel_lambda*line_wave
            lam_sigma = rel_lambda_sigma*line_wave

            model += self._gas_line_model(wave, line_flux, lam, lam_sigma, line_wave)

            # Add coupled lines
            nclines = len(line)//2
            for j in range(nclines):
                line_wave = line[1+2*j]
                factor = line[2+2*j]
                lam = rel_lambda*line_wave
                lam_sigma = rel_lambda_sigma*line_wave
                model += self._gas_line_model(
                    wave, factor*line_flux, lam, lam_sigma, line_wave)

        return model

    def _gas_line_model(self, wave, flux, lam, lam_sigma, line_wave_rest):
        """Gas emission line model.

        Parameters
        ----------
        wave : array-like
            Wavelength array.
        flux : float
            Line flux.
        lam : float
            Line wavelength.
        lam_sigma : float
            Line standard deviation.
        line_wave_rest : float
            Rest wavelength of the line (for LSF lookup).

        Returns
        -------
        gas_line_model : np.ndarray
        """
        # Get window-specific LSF sigma for this line
        lsf_sigma = self.get_lsf_sigma_for_wavelength(line_wave_rest)
        
        # Handle potential irregular wavelength spacing
        if len(wave) == 1:
            dwave = 1.0  # Single wavelength point
        else:
            # Calculate local wavelength spacing
            dwave = np.zeros(len(wave))
            
            # First bin
            if len(wave) > 1:
                dwave[0] = wave[1] - wave[0]
            else:
                dwave[0] = 1.0
                
            # Middle bins
            for i in range(1, len(wave) - 1):
                dwave[i] = 0.5 * (wave[i+1] - wave[i-1])
                
            # Last bin
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
        """Baseline model.

        Baseline is assumed to follow a polynomial function of order
        self.baseline_order. if self.baseline_order is None this function is
        not called.

        Parameters
        ----------
        wave : array-like
            Wavelength array.
        baseline_param : array-like
            List of coefficients for polynomial model.

        Returns
        -------
        baseline : np.ndarray
        """
        wave_shft = wave - self.wave_ref
        baseline = np.ones(len(wave))*baseline_param[0]
        for i in range(self.baseline_order):
            baseline += baseline_param[1+i]*wave_shft**(1+i)

        return baseline