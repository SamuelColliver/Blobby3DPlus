"""Metadata.

Organisation of the Blobby3D metadata that describes the coordinates with
support for multiple wavelength windows.

@original author: Mathew Varidel
@edited by: Samuel Colliver
"""

from pathlib import Path
import numpy as np


class WavelengthWindow:
    """Container for wavelength window information."""
    def __init__(self, r_min, r_max, orig_start_bin, orig_end_bin, n_bins):
        self.r_min = r_min
        self.r_max = r_max
        self.orig_start_bin = orig_start_bin
        self.orig_end_bin = orig_end_bin
        self.n_bins = n_bins
        self.dr = (r_max - r_min) / n_bins
        
        # These will be set during processing
        self.start_idx = None
        self.end_idx = None
        self.r = None


class Metadata:

    def __init__(self, metadata_path):
        """Metadata organisation object.

        Parameters
        ----------
        metadata_path : str or pathlib.Path

        Returns
        -------
        None.

        Attributes
        ----------
        naxis : np.ndarray
            Array dimensions [ni, nj, nr_total]
        sz : int
            Total size of the data cube
        x_lim : tuple
            Left and right boundaries of the x axis
        y_lim : tuple  
            Bottom and top boundaries of the y axis
        dx : float
            Width of pixels along the x axis
        dy : float
            Width of pixels along the y axis
        dr : float
            Average wavelength bin width
        wavelength_windows : list
            List of WavelengthWindow objects
        r_full : np.ndarray
            Combined wavelength array from all windows
        """
        self.wavelength_windows = []
        self.r_full = None
        
        # Load metadata from file
        self._load_metadata(Path(metadata_path))
        
        # Process wavelength windows
        self._process_wavelength_windows()
        
        # Set legacy attributes for backwards compatibility
        self.naxis = np.array([self.ni, self.nj, self.nr], dtype=int)
        self.sz = self.naxis.prod()
        self.x_lim = (self.x_min, self.x_max)
        self.y_lim = (self.y_min, self.y_max)
        self.dx = (self.x_max - self.x_min) / self.nj
        self.dy = (self.y_max - self.y_min) / self.ni
        
        # Calculate average dr for legacy compatibility
        if self.wavelength_windows:
            total_range = self.wavelength_windows[-1].r_max - self.wavelength_windows[0].r_min
            self.dr = total_range / self.nr
        else:
            self.dr = 1.0  # Fallback value

    def _load_metadata(self, metadata_path):
        """Load metadata from the new format file."""
        print(f"Loading metadata from: {metadata_path}")
        
        # Initialisation flags
        ni_set = nj_set = False
        x_min_set = x_max_set = y_min_set = y_max_set = False
        
        try:
            with open(metadata_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    # Skip empty lines and comments
                    if not line.strip() or line.strip().startswith('#'):
                        continue
                    
                    # Remove inline comments
                    if '#' in line:
                        line = line[:line.index('#')]
                    
                    # Parse line
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    keyword = parts[0].upper()
                    
                    try:
                        if keyword == 'NI':
                            self.ni = int(parts[1])
                            ni_set = True
                            print(f"  NI: {self.ni}")
                            
                        elif keyword == 'NJ':
                            self.nj = int(parts[1])
                            nj_set = True
                            print(f"  NJ: {self.nj}")
                            
                        elif keyword == 'X_MIN':
                            self.x_min = float(parts[1])
                            x_min_set = True
                            print(f"  X_MIN: {self.x_min}")
                            
                        elif keyword == 'X_MAX':
                            self.x_max = float(parts[1])
                            x_max_set = True
                            print(f"  X_MAX: {self.x_max}")
                            
                        elif keyword == 'Y_MIN':
                            self.y_min = float(parts[1])
                            y_min_set = True
                            print(f"  Y_MIN: {self.y_min}")
                            
                        elif keyword == 'Y_MAX':
                            self.y_max = float(parts[1])
                            y_max_set = True
                            print(f"  Y_MAX: {self.y_max}")
                            
                        elif keyword == 'WAVE_RANGE':
                            if len(parts) < 6:
                                raise ValueError(f"WAVE_RANGE requires 5 values, got {len(parts)-1}")
                            
                            r_min = float(parts[1])
                            r_max = float(parts[2])
                            orig_start = int(parts[3])
                            orig_end = int(parts[4])
                            n_bins = int(parts[5])
                            
                            # Validation
                            if r_min >= r_max:
                                raise ValueError(f"Invalid wavelength range: {r_min} >= {r_max}")
                            if orig_start > orig_end:
                                raise ValueError(f"Invalid bin range: {orig_start} > {orig_end}")
                            if n_bins != (orig_end - orig_start + 1):
                                raise ValueError(f"Inconsistent bin count: {n_bins} != {orig_end - orig_start + 1}")
                            
                            window = WavelengthWindow(r_min, r_max, orig_start, orig_end, n_bins)
                            self.wavelength_windows.append(window)
                            
                            print(f"  WAVE_RANGE: {r_min} - {r_max} Å "
                                  f"(bins {orig_start}-{orig_end}, n={n_bins})")
                                  
                        elif keyword and not keyword.startswith('#'):
                            print(f"  WARNING: Unknown keyword '{keyword}' on line {line_num}")
                            
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Error parsing line {line_num}: '{line.strip()}' - {e}")
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading metadata: {e}")
        
        # Validate required parameters
        if not ni_set:
            raise ValueError("'ni' not specified in metadata file")
        if not nj_set:
            raise ValueError("'nj' not specified in metadata file")
        if not (x_min_set and x_max_set and y_min_set and y_max_set):
            raise ValueError("Spatial coordinates not fully specified in metadata file")
        if not self.wavelength_windows:
            raise ValueError("No wavelength ranges specified in metadata file")
        
        # Sort windows by start bin
        self.wavelength_windows.sort(key=lambda w: w.orig_start_bin)
        
        # Validate sequential bins
        self._validate_bin_indices()
        
        print("Metadata loaded successfully.")
        print(f"Found {len(self.wavelength_windows)} wavelength windows")

    def _validate_bin_indices(self):
        """Validate that bin indices are sequential and non-overlapping."""
        expected_start = 0
        
        for i, window in enumerate(self.wavelength_windows):
            if window.orig_start_bin != expected_start:
                raise ValueError(
                    f"Non-sequential bin indices in window {i+1}: "
                    f"expected {expected_start}, got {window.orig_start_bin}"
                )
            expected_start = window.orig_end_bin + 1
            print(f"  Validated window {i+1}: bins {window.orig_start_bin}-{window.orig_end_bin}")

    def _process_wavelength_windows(self):
        """Process wavelength windows and create combined arrays."""
        total_bins = 0
        
        for i, window in enumerate(self.wavelength_windows):
            # Set global indices
            window.start_idx = total_bins
            window.end_idx = total_bins + window.n_bins - 1
            total_bins += window.n_bins
            
            # Create wavelength array for this window
            window.r = np.linspace(
                window.r_min + 0.5 * window.dr,
                window.r_max - 0.5 * window.dr,
                window.n_bins
            )
            
            print(f"  Window {i+1}: [{window.r_min}, {window.r_max}] Å "
                  f"({window.n_bins} bins, dr={window.dr:.4f} Å/bin)")
        
        self.nr = total_bins
        
        # Create combined wavelength array
        self.r_full = np.concatenate([window.r for window in self.wavelength_windows])
        
        print(f"Total wavelength coverage: {self.wavelength_windows[0].r_min} - "
              f"{self.wavelength_windows[-1].r_max} Å")
        print(f"Total wavelength bins: {self.nr}")

    def get_axis_array(self, axis):
        """Get axis array.

        Calculate the x, y or wavelength array.

        Parameters
        ----------
        axis : str
            Axis defined by 'x', 'y', 'r'. Where 'r' is the wavelength.

        Returns
        -------
        axis_array : np.ndarray
        """
        if axis == 'x':
            n = self.nj
            lim_min, lim_max = self.x_min, self.x_max
            d = self.dx
        elif axis == 'y':
            n = self.ni
            lim_min, lim_max = self.y_min, self.y_max
            d = self.dy
        elif axis == 'r':
            return self.r_full.copy()
        else:
            raise ValueError(f"Unknown axis: {axis}. Must be 'x', 'y', or 'r'")

        return np.linspace(lim_min + 0.5*d, lim_max - 0.5*d, n)

    def get_window_for_wavelength(self, wavelength):
        """Find which wavelength window contains a given wavelength.
        
        Parameters
        ----------
        wavelength : float
            Wavelength in Angstroms
            
        Returns
        -------
        window : WavelengthWindow or None
            The window containing the wavelength, or None if not found
        """
        for window in self.wavelength_windows:
            if window.r_min <= wavelength <= window.r_max:
                return window
        return None

    def get_global_wavelength_index(self, wavelength):
        """Get the global wavelength index for a given wavelength.
        
        Parameters
        ----------
        wavelength : float
            Wavelength in Angstroms
            
        Returns
        -------
        index : int or None
            Global index in the combined wavelength array, or None if not found
        """
        window = self.get_window_for_wavelength(wavelength)
        if window is None:
            return None
            
        # Find local index within the window
        local_idx = np.argmin(np.abs(window.r - wavelength))
        return window.start_idx + local_idx

    # Legacy methods for backwards compatibility
    @property
    def r_lim(self):
        """Legacy attribute: wavelength range."""
        if self.wavelength_windows:
            return (self.wavelength_windows[0].r_min, self.wavelength_windows[-1].r_max)
        return (0, 1)