#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct comparison plots.

@author: Mathew Varidel
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_comparison_maps(
        comp_shape, map_shape, figsize,
        right_pad=0.1, left_pad=0.05, bottom_pad=0.05,
        xlabel=r'$\Delta$RA ($^{\prime\prime}$)',
        ylabel=r'$\Delta$Dec ($^{\prime\prime}$)'):
    """
    Setup comparison plot.

    This will setup a number of maps of shape 'comp_shape' plus a set of
    residual maps. This is a little fiddly, you may need to play around with
    the pad and figsize attributes to get everything to sit where you want.

    Parameters
    ----------
    comp_shape : list of int
        Comparison map shape (rows, columns). Each row represents a quantity
        of interest, each column represents a map of that quantity for
        comparison.
    map_shape : list of int
        Map array shape in (rows, columns).
    figsize : width, height in inches
        This needs to be defined in order to plot maps with given axis ratio.
    right_pad : float
    left_pad : float
    bottom_pad : float
    xlabel : str
    ylabel : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : nested list of matplotlib.axes._subplots.AxesSubplot
    """
    fig = plt.figure(figsize=figsize)

    maps_tot = comp_shape[1] + 1

    gs_im1 = mpl.gridspec.GridSpec(*comp_shape)
    gs_cb1 = mpl.gridspec.GridSpec(comp_shape[0], 1)
    gs_res = mpl.gridspec.GridSpec(comp_shape[0], 1)
    gs_cb2 = mpl.gridspec.GridSpec(comp_shape[0], 1)

    left_pad = left_pad
    right_pad = right_pad
    bottom_pad = bottom_pad
    cb_width = 0.03
    cb1_right = 0.1
    hspace = 0.0
    wspace = 0.0
    rem_width = (
            1.0 - left_pad - right_pad
            - 2*cb_width - cb1_right - maps_tot*wspace
            )

    im_width = np.round(rem_width/maps_tot, 3)

    # below sets the top such that the aspect ratio for images are correct
    fig_width = fig.get_size_inches()[0]
    im_abswidth = fig_width*im_width

    aim_im_absheight = im_abswidth*map_shape[0]/map_shape[1]

    fig_height = fig.get_size_inches()[1]
    im_height = np.round(aim_im_absheight/fig_height, 3)
    top = comp_shape[0]*im_height + bottom_pad

    # model+data images
    left = left_pad
    right = left + im_width*comp_shape[1] + wspace*comp_shape[1]
    gs_im1.update(
            left=left,
            right=right,
            wspace=wspace,
            bottom=bottom_pad,
            top=top,
            hspace=hspace
            )

    # model+data colorbar
    left = right
    right = left + cb_width
    gs_cb1.update(
            left=left,
            right=right,
            bottom=bottom_pad,
            top=top,
            hspace=hspace
            )

    # residual image
    left = right + cb1_right
    right = left + im_width + wspace
    gs_res.update(
            left=left,
            right=right,
            bottom=bottom_pad,
            top=top,
            hspace=hspace
            )

    # residual colorbar
    left = right
    right = left + cb_width
    gs_cb2.update(
            left=left,
            right=right,
            bottom=bottom_pad,
            top=top,
            hspace=hspace
            )

    ax = []
    for i in range(comp_shape[0]):
        ax_tmp = []

        # Add map axes
        for m in range(comp_shape[1]):
            ax_tmp.append(plt.subplot(gs_im1[i, m]))
        ax_tmp.append(plt.subplot(gs_cb1[i, 0]))

        # Add residual axes
        ax_tmp.append(plt.subplot(gs_res[i, 0]))
        ax_tmp.append(plt.subplot(gs_cb2[i, 0]))

        ax.append(ax_tmp)

    # Plot spatial titles
    fig.text(
        left_pad + 0.5*comp_shape[1]*im_width,
        0.0*bottom_pad,
        xlabel,
        ha='center', fontsize='large'
        )

    fig.text(
        left_pad + (comp_shape[1] + 0.5)*im_width + cb_width + cb1_right,
        0.0*bottom_pad,
        xlabel,
        ha='center', fontsize='large'
        )

    fig.text(
        0.0,
        bottom_pad + 0.5*comp_shape[0]*im_height,
        ylabel,
        va='center', rotation='vertical', fontsize='large'
        )

    return fig, ax


def map_limits(data, pct=100.0, vlim=None, absolute=False):
    """
    Get limits for maps.

    Parameters
    ----------
    data : 2D numpy.array or list of 2D numpy.array
        Map data or list of map data
    pct : float, default 95%
        Percentile to use to calculate limit.
    vlim : list of float, default None
        Absolute limits applied to map.
    absolute : bool, default False
        Whether to calculate limits about 0.
    """
    if isinstance(data, list):
        data_fin = []
        for d in data:
            data_fin.append(d[np.isfinite(d)])
        data_fin = np.array(data_fin)
    else:
        data_fin = data[np.isfinite(data)]

    if vlim is not None:
        data_lim = vlim
    elif not absolute:
        data_lim = [
                np.nanpercentile(data_fin, 100.0 - pct),
                np.nanpercentile(data_fin, pct)
                ]
    else:
        data_abs = np.absolute(data)
        data_lim_max = np.nanpercentile(data_abs, pct)
        data_lim = [-data_lim_max, data_lim_max]

    return data_lim


def colorbar(
        mappable, cax, clim, label=None, label_fontsize='large',
        numticks=5, tick_fontsize='large', tick_direction='in', tick_pad=4.0):
    """
    Plot colorbar on a given axis.

    """
    mappable.set_clim(*clim)
    cb = plt.colorbar(mappable, cax=cax)

    if isinstance(label, str):
        cb.set_label(label=label, fontsize=label_fontsize)

    cb.locator = mpl.ticker.LinearLocator(numticks=numticks)
    cb.ax.tick_params(
            labelsize=tick_fontsize,
            direction=tick_direction,
            pad=tick_pad)
