#!/usr/bin/env python3
"""
Visualization functions for IFS data processing.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_comparison_plot(original_data, windowed_data, metadata, windows, output_dir):
    """
    Create comparison plots showing original vs windowed data.
    
    Parameters
    ----------
    original_data : np.ndarray
        Original data cube in (ni, nj, nr) format
    windowed_data : np.ndarray
        Windowed data cube in (ni, nj, nr_windowed) format
    metadata : dict
        Metadata dictionary
    windows : list
        List of window dictionaries
    output_dir : str or Path
        Output directory for plots
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # original wavelengths
    orig_wave = metadata['wavelengths']
    wave_max = np.max(orig_wave)
    wave_min = np.min(orig_wave)
    
    # windowed wavelengths
    windowed_wave = []
    for window in windows:
        start_idx = window['start_idx']
        end_idx = window['end_idx']
        window_wave = orig_wave[start_idx:end_idx+1]
        windowed_wave.extend(window_wave)
    windowed_wave = np.array(windowed_wave)
    
    # spatial coordinates
    x_extent = [metadata['x_min'], metadata['x_max']]
    y_extent = [metadata['y_min'], metadata['y_max']]
    
    # 1. total flux maps
    orig_flux = np.sum(original_data, axis=2)
    windowed_flux = np.sum(windowed_data, axis=2)
    
    im1 = axes[0,0].imshow(orig_flux, origin='lower', aspect='auto', 
                          extent=x_extent + y_extent)
    axes[0,0].set_title('Original Total Flux Map')
    axes[0,0].set_xlabel('X (arcsec)')
    axes[0,0].set_ylabel('Y (arcsec)')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(windowed_flux, origin='lower', aspect='auto',
                          extent=x_extent + y_extent)
    axes[0,1].set_title('Windowed Total Flux Map')
    axes[0,1].set_xlabel('X (arcsec)')
    axes[0,1].set_ylabel('Y (arcsec)')
    plt.colorbar(im2, ax=axes[0,1])
    
    # 2. central spectra
    center_i, center_j = metadata['ni']//2, metadata['nj']//2
    orig_spectrum = original_data[center_i, center_j, :]
    windowed_spectrum = windowed_data[center_i, center_j, :]
    
    axes[1,0].plot(orig_wave, orig_spectrum, 'b-', linewidth=1, label='Original')
    axes[1,0].scatter(orig_wave, orig_spectrum, s=2, alpha=0.7) 
    axes[1,0].set_xlim(wave_min, wave_max)
    axes[1,0].set_title(f'Original Central Spectrum\n(spaxel {center_i}, {center_j})')
    axes[1,0].set_xlabel('Wavelength (Å)')
    axes[1,0].set_ylabel('Flux')
    axes[1,0].grid(True, alpha=0.3)
    
    # add window regions to original spectrum
    colors = plt.cm.Set3(np.linspace(0, 1, len(windows)))
    for i, window in enumerate(windows):
        axes[1,0].axvspan(window['actual_r_min'], window['actual_r_max'], 
                         alpha=0.3, color=colors[i], label=window['name'])
    
    # only show legend if we have few windows to avoid clutter
    if len(windows) <= 5:
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    axes[1,1].plot(windowed_wave, windowed_spectrum, 'r-', linewidth=1, label='Windowed')
    axes[1,1].scatter(windowed_wave, windowed_spectrum, s=2, alpha=0.7)
    axes[1,1].set_xlim(wave_min, wave_max)
    axes[1,1].set_title(f'Windowed Central Spectrum\n(spaxel {center_i}, {center_j})')
    axes[1,1].set_xlabel('Wavelength (Å)')
    axes[1,1].set_ylabel('Flux')
    axes[1,1].grid(True, alpha=0.3)
    
    # 3. integrated spectra
    orig_total = np.sum(original_data, axis=(0,1))
    windowed_total = np.sum(windowed_data, axis=(0,1))
    
    axes[2,0].plot(orig_wave, orig_total, 'b-', linewidth=1)
    axes[2,0].scatter(orig_wave, orig_total, s=2, alpha=0.7)
    axes[2,0].set_xlim(wave_min, wave_max)
    axes[2,0].set_title('Original Integrated Spectrum')
    axes[2,0].set_xlabel('Wavelength (Å)')
    axes[2,0].set_ylabel('Total Flux')
    axes[2,0].grid(True, alpha=0.3)
    
    # add window regions
    for i, window in enumerate(windows):
        axes[2,0].axvspan(window['actual_r_min'], window['actual_r_max'], 
                         alpha=0.3, color=colors[i])
    
    axes[2,1].plot(windowed_wave, windowed_total, 'r-', linewidth=1)
    axes[2,1].scatter(windowed_wave, windowed_total, s=2, alpha=0.7)
    axes[2,1].set_xlim(wave_min, wave_max)
    axes[2,1].set_title('Windowed Integrated Spectrum')
    axes[2,1].set_xlabel('Wavelength (Å)')
    axes[2,1].set_ylabel('Total Flux')
    axes[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save plot
    plot_file = Path(output_dir) / 'windowed_data_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Comparison plot: {plot_file}")
    
    return fig


def create_multi_arm_comparison_plot(individual_results, combined_result, windowed_data, windows, output_dir):
    """
    Create comparison plots for multi-arm data (e.g., blue + red SAMI).
    
    Parameters
    ----------
    individual_results : list
        List of individual arm data results
    combined_result : dict
        Combined data result
    windowed_data : np.ndarray
        Final windowed data
    windows : list
        List of window dictionaries
    output_dir : str or Path
        Output directory for plots
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    n_arms = len(individual_results)
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    
    # get spatial extents
    coord_info = combined_result['coord_info']
    x_extent = [coord_info['x_min'], coord_info['x_max']]
    y_extent = [coord_info['y_min'], coord_info['y_max']]
    
    # wavelength arrays
    combined_waves = combined_result['wavelengths']
    
    # windowed wavelengths
    windowed_waves = []
    for window in windows:
        start_idx = window['start_idx']
        end_idx = window['end_idx']
        window_waves = combined_waves[start_idx:end_idx+1]
        windowed_waves.extend(window_waves)
    windowed_waves = np.array(windowed_waves)
    
    # 1. flux maps for each arm
    if n_arms >= 2:
        arm1_flux = np.nansum(individual_results[0]['data'], axis=0)
        arm2_flux = np.nansum(individual_results[1]['data'], axis=0)
        
        im1 = axes[0,0].imshow(arm1_flux, origin='lower', aspect='auto', extent=x_extent + y_extent)
        axes[0,0].set_title(f'Arm 1 Total Flux Map')
        axes[0,0].set_xlabel('X (arcsec)')
        axes[0,0].set_ylabel('Y (arcsec)')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].imshow(arm2_flux, origin='lower', aspect='auto', extent=x_extent + y_extent)
        axes[0,1].set_title(f'Arm 2 Total Flux Map')
        axes[0,1].set_xlabel('X (arcsec)')
        axes[0,1].set_ylabel('Y (arcsec)')
        plt.colorbar(im2, ax=axes[0,1])
    
    # 2. combined and windowed flux maps
    combined_flux = np.nansum(combined_result['data'], axis=0)
    windowed_flux = np.nansum(windowed_data, axis=2)
    
    im3 = axes[1,0].imshow(combined_flux, origin='lower', aspect='auto', extent=x_extent + y_extent)
    axes[1,0].set_title('Combined Total Flux Map')
    axes[1,0].set_xlabel('X (arcsec)')
    axes[1,0].set_ylabel('Y (arcsec)')
    plt.colorbar(im3, ax=axes[1,0])
    
    im4 = axes[1,1].imshow(windowed_flux, origin='lower', aspect='auto', extent=x_extent + y_extent)
    axes[1,1].set_title('Windowed Total Flux Map')
    axes[1,1].set_xlabel('X (arcsec)')
    axes[1,1].set_ylabel('Y (arcsec)')
    plt.colorbar(im4, ax=axes[1,1])
    
    # 3. central spectra for individual arms
    ni, nj = combined_result['data'].shape[1], combined_result['data'].shape[2]
    center_i, center_j = ni//2, nj//2
    
    if n_arms >= 2:
        arm1_spectrum = individual_results[0]['data'][:, center_i, center_j]
        arm2_spectrum = individual_results[1]['data'][:, center_i, center_j]
        arm1_waves = individual_results[0]['wavelengths']
        arm2_waves = individual_results[1]['wavelengths']
        
        axes[2,0].plot(arm1_waves, arm1_spectrum, 'b-', linewidth=1, label='Arm 1')
        axes[2,0].set_title(f'Arm 1 Central Spectrum\n(spaxel {center_i}, {center_j})')
        axes[2,0].set_xlabel('Wavelength (Å)')
        axes[2,0].set_ylabel('Flux')
        axes[2,0].grid(True, alpha=0.3)
        
        axes[2,1].plot(arm2_waves, arm2_spectrum, 'r-', linewidth=1, label='Arm 2')
        axes[2,1].set_title(f'Arm 2 Central Spectrum\n(spaxel {center_i}, {center_j})')
        axes[2,1].set_xlabel('Wavelength (Å)')
        axes[2,1].set_ylabel('Flux')
        axes[2,1].grid(True, alpha=0.3)
    
    # 4. combined spectrum with windows and final windowed spectrum
    combined_spectrum = combined_result['data'][:, center_i, center_j]
    windowed_spectrum = windowed_data[center_i, center_j, :]
    
    axes[3,0].plot(combined_waves, combined_spectrum, 'k-', linewidth=1, label='Combined')
    axes[3,0].set_title(f'Combined Central Spectrum\n(spaxel {center_i}, {center_j})')
    axes[3,0].set_xlabel('Wavelength (Å)')
    axes[3,0].set_ylabel('Flux')
    axes[3,0].grid(True, alpha=0.3)
    
    # add window regions
    colors = plt.cm.Set3(np.linspace(0, 1, len(windows)))
    for i, window in enumerate(windows):
        axes[3,0].axvspan(window['actual_r_min'], window['actual_r_max'], 
                         alpha=0.3, color=colors[i], label=window['name'])
    
    # only show legend if we have few windows
    if len(windows) <= 5:
        axes[3,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # plot windowed spectrum
    axes[3,1].plot(windowed_waves, windowed_spectrum, 'g-', linewidth=1, label='Windowed')
    axes[3,1].scatter(windowed_waves, windowed_spectrum, s=10, alpha=0.7)
    axes[3,1].set_title(f'Windowed Central Spectrum\n(spaxel {center_i}, {center_j})')
    axes[3,1].set_xlabel('Wavelength (Å)')
    axes[3,1].set_ylabel('Flux')
    axes[3,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save plot
    plot_file = Path(output_dir) / 'multi_arm_windowed_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Multi-arm comparison plot: {plot_file}")
    
    return fig


def create_processing_summary_plot(data_result, windows, output_dir):
    """
    Create a summary plot showing the data processing results.
    
    Parameters
    ----------
    data_result : dict
        Processed data result dictionary
    windows : list
        List of window dictionaries
    output_dir : str or Path
        Output directory for plots
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # get coordinate info
    coord_info = data_result['coord_info']
    x_extent = [coord_info['x_min'], coord_info['x_max']]
    y_extent = [coord_info['y_min'], coord_info['y_max']]
    
    # 1. total flux map
    total_flux = np.nansum(data_result['data'], axis=0)
    im1 = axes[0,0].imshow(total_flux, origin='lower', aspect='auto', 
                          extent=x_extent + y_extent)
    axes[0,0].set_title('Total Flux Map')
    axes[0,0].set_xlabel('X (arcsec)')
    axes[0,0].set_ylabel('Y (arcsec)')
    plt.colorbar(im1, ax=axes[0,0])
    
    # 2. variance map (if available)
    if data_result['variance'] is not None:
        total_var = np.nansum(data_result['variance'], axis=0)
        # use log scale for variance
        im2 = axes[0,1].imshow(np.log10(total_var + 1e-20), origin='lower', aspect='auto',
                              extent=x_extent + y_extent)
        axes[0,1].set_title('Log Total Variance Map')
        axes[0,1].set_xlabel('X (arcsec)')
        axes[0,1].set_ylabel('Y (arcsec)')
        plt.colorbar(im2, ax=axes[0,1])
    else:
        axes[0,1].text(0.5, 0.5, 'No Variance Data', transform=axes[0,1].transAxes,
                      ha='center', va='center', fontsize=14)
        axes[0,1].set_title('Variance Map (N/A)')
    
    # 3. spectral coverage
    wavelengths = data_result['wavelengths']
    total_spectrum = np.nansum(data_result['data'], axis=(1,2))
    
    axes[1,0].plot(wavelengths, total_spectrum, 'b-', linewidth=1)
    axes[1,0].set_title('Total Integrated Spectrum')
    axes[1,0].set_xlabel('Wavelength (Å)')
    axes[1,0].set_ylabel('Total Flux')
    axes[1,0].grid(True, alpha=0.3)
    
    # highlight windows if provided
    if windows:
        colors = plt.cm.Set3(np.linspace(0, 1, len(windows)))
        for i, window in enumerate(windows):
            if 'actual_r_min' in window and 'actual_r_max' in window:
                axes[1,0].axvspan(window['actual_r_min'], window['actual_r_max'], 
                                 alpha=0.3, color=colors[i])
    
    # 4. signal-to-noise ratio
    if data_result['variance'] is not None:
        # calculate S/N ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = data_result['data'] / np.sqrt(data_result['variance'])
            snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
        
        median_snr = np.nanmedian(snr, axis=0)
        im4 = axes[1,1].imshow(median_snr, origin='lower', aspect='auto',
                              extent=x_extent + y_extent)
        axes[1,1].set_title('Median S/N Ratio Map')
        axes[1,1].set_xlabel('X (arcsec)')
        axes[1,1].set_ylabel('Y (arcsec)')
        plt.colorbar(im4, ax=axes[1,1])
    else:
        axes[1,1].text(0.5, 0.5, 'No S/N Data', transform=axes[1,1].transAxes,
                      ha='center', va='center', fontsize=14)
        axes[1,1].set_title('S/N Ratio Map (N/A)')
    
    plt.tight_layout()
    
    # save plot
    plot_file = Path(output_dir) / 'processing_summary.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Processing summary plot: {plot_file}")
    
    return fig