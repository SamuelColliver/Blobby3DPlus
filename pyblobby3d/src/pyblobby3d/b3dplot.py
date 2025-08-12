#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting routines.

@author: Mathew Varidel

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class cmap:
    flux = 'Greys_r'
    v = 'RdYlBu_r'
    vdisp = 'YlOrBr'
    residuals = 'RdYlBu_r'


def plot_map(
        ax, map_2d,
        title=None,
        xlabel=None,
        ylabel=None,
        xticks=False,
        yticks=False,
        xlim=None,
        ylim=None,
        colorbar=False,
        colorbar_cax=None,
        cbar_label=None,
        clim=None,
        cbarticks=None,
        cbar_nticks=5,
        logscale=False,
        cmap=None,
        title_fontsize='large',
        label_fontsize='large',
        tick_fontsize='large',
        cb_label_fontsize='large',
        cb_tick_fontsize='large',
        fwhm=None,
        aspect=None,
        mask=None):
    """Plot singular 2d map to an axes."""
    naxis = map_2d.shape

    if clim is not None:
        if clim[0] is None:
            clim[0] = np.nanmin(map_2d)
        if clim[1] is None:
            clim[1] = np.nanmax(map_2d)
    else:
        clim = [np.nanmin(map_2d), np.nanmax(map_2d)]

    if logscale:
        norm = mpl.colors.LogNorm(vmin=clim[0], vmax=clim[1])
    else:
        norm = None

    ax.imshow(
        map_2d,
        origin='lower',
        interpolation='nearest',
        norm=norm,
        vmin=clim[0], vmax=clim[1],
        cmap=cmap,
        aspect=aspect
        )

    ax.tick_params(axis='both', direction='in', pad=4.0)

    if isinstance(title, str):
        ax.set_title(title, fontsize=title_fontsize)

    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel, fontsize=label_fontsize)

    if isinstance(ylabel, str):
        ax.set_ylabel(ylabel, fontsize=label_fontsize)

    set_ticks(
            ax.xaxis,
            ticks=xticks,
            naxis=naxis[1],
            lim=xlim,
            tick_fontsize=tick_fontsize)

    set_ticks(
            ax.yaxis,
            ticks=yticks,
            naxis=naxis[0],
            lim=ylim,
            tick_fontsize=tick_fontsize)



    # XTicks
#    # TODO: Combine XTicks/YTicks settings
#    if not xticks:
#        ax.set_xticks([])
#    elif xticks == 'all':
#        x_dist = np.diff(self.x_lim)[0]/2.0
#        ax.set_xticks([-0.5, (naxis[1]-1.0)/2.0, naxis[1]-0.5])
#        xtick_values = [-round(x_dist, 1), 0.0, round(x_dist, 1)]
#        ax.set_xticklabels(xtick_values, fontsize=tick_fontsize)
#    elif xticks == 'upper':
#        x_dist = np.diff(self.x_lim)[0]/2.0
#        ax.set_xticks([(naxis[1]-1.0)/2.0, naxis[1]-0.5])
#        xtick_values = [0.0, round(x_dist, 1)]
#        ax.set_xticklabels(xtick_values, fontsize=tick_fontsize)
#    elif xticks == 'lower':
#        x_dist = np.diff(self.x_lim)[0]/2.0
#        ax.set_xticks([-0.5, (naxis[1]-1.0)/2.0])
#        xtick_values = [-round(x_dist, 1), 0.0]
#        ax.set_xticklabels(xtick_values, fontsize=tick_fontsize)

#    # YTicks
#    if not yticks:
#        ax.set_yticks([])
#    elif yticks == 'all':
#        y_dist = np.diff(self.y_lim)[0]/2.0
#        ax.set_yticks([-0.5, (naxis[0]-1.0)/2.0, naxis[0]-0.5])
#        ytick_values = [-round(y_dist, 1), 0.0, round(y_dist, 1)]
#        ax.set_yticklabels(ytick_values, fontsize=tick_fontsize)
#    elif yticks == 'upper':
#        y_dist = np.diff(self.y_lim)[0]/2.0
#        ax.set_yticks([(naxis[0]-1.0)/2.0, naxis[0]-0.5])
#        ytick_values = [0.0, round(y_dist, 1)]
#        ax.set_yticklabels(ytick_values, fontsize=tick_fontsize)
#    elif yticks == 'lower':
#        y_dist = np.diff(self.y_lim)[0]/2.0
#        ax.set_yticks([-0.5, (naxis[0]-1.0)/2.0])
#        ytick_values = [-round(y_dist, 1), 0.0]
#        ax.set_yticklabels(ytick_values, fontsize=tick_fontsize)

    if colorbar:
        if colorbar_cax is None:
            divider = make_axes_locatable(ax)
            colorbar_cax = divider.append_axes(
                    'right', size='10%', pad=0.03)

        plot_colorbar(
                cax=colorbar_cax,
                clim=clim,
                label=cbar_label,
                label_fontsize=cb_label_fontsize,
                cbarticks=cbarticks,
                cbar_nticks=cbar_nticks,
                tick_fontsize=cb_tick_fontsize,
                cmap=cmap
                )

    if fwhm is not None:
        circle = plt.Circle(
                (1.1*fwhm, 1.1*fwhm),
                radius=fwhm,
                fill=False, edgecolor='r', linewidth=1.0)
        ax.add_artist(circle)


def set_ticks(ax, naxis, ticks=False, lim=None, tick_fontsize='large'):
    if not ticks:
        ax.set_ticks([])
    elif ticks == 'all':
        dist = np.diff(lim)[0]/2.0
        ax.set_ticks([-0.5, (naxis-1.0)/2.0, naxis-0.5])
        tick_values = [-round(dist, 1), 0.0, round(dist, 1)]
        ax.set_ticklabels(tick_values, fontsize=tick_fontsize)
    elif ticks == 'upper':
        dist = np.diff(lim)[0]/2.0
        ax.set_ticks([(naxis-1.0)/2.0, naxis-0.5])
        tick_values = [0.0, round(dist, 1)]
        ax.set_ticklabels(tick_values, fontsize=tick_fontsize)
    elif ticks == 'lower':
        dist = np.diff(lim)[0]/2.0
        ax.set_ticks([-0.5, (naxis-1.0)/2.0])
        tick_values = [-round(dist, 1), 0.0]
        ax.set_ticklabels(tick_values, fontsize=tick_fontsize)
    else:
        raise ValueError('xticks must be False, all, upper, lower.')


def plot_colorbar(
        cax,
        mappable=None,
        label=None,
        xlabel=None,
        ylabel=None,
        clim=None,
        cbarticks=None,
        cbar_nticks=5,
        logscale=False,
        cmap=None,
        label_fontsize='large',
        tick_fontsize='large'):
    """
    Plot colorbar.

    Parameters
    ----------
    cax : matplotlib.pyplot.axes
        Colorbar axis.
    mappable : matplotlib.cm.ScalarMappable, default None
        Mappable object -- usually image. Default None uses a linear
        mappable between limits constructed by clim.
    title : str, default None
    xlabel : str, default None
    ylabel : str, default None
    clim : list, default None
        Color limits. Default uses mappable to construct color limits.
    cbarticks : list, default None
        cbarticks[0] corresponds to ticks.
        cbarticks[1] corresponds to tick labels.
    cbar_nticks : int, default 5
        Number of colorbar ticks if cbarticks is None.
    logscale : bool, default False
        Not allowed at this time.
    cmap : Matplotlib.colormap instance
    label_fontsize : matplotlib fontsize property
    tick_fontisze : matplotlib fontsize property
    """
    assert (
            (mappable is not None)
            or (mappable is None and clim is not None)
            )

    # Deal with limits
    if clim is not None:
        if clim[0] is None:
            clim[0] = np.nanmin(mappable)
        if clim[1] is None:
            clim[1] = np.nanmax(mappable)
    else:
        clim = [
                np.nanmin(mappable),
                np.nanmax(mappable)
                ]

    # logscale
    if logscale:
        cb_norm = mpl.colors.LogNorm(vmin=clim[0], vmax=clim[1])

        # tick formatter
        def logformatter(x, pos):
            value = np.exp(np.log(clim[0]) + x*np.log(clim[1]/clim[0]))
            value = round(value, -int(np.floor(np.log10(abs(value)))))
            return '%i' % (value)
    else:
        cb_norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])

    # Construct default mappable
    if mappable is None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=cb_norm)
        sm.set_array([])

    # Construct colorbar
    cb = plt.colorbar(sm, cax=cax)

    if label is not None:
        cb.set_label(label=label, fontsize=label_fontsize)

    cb.set_clim(clim[0], clim[1])

    # Set ticks
    if isinstance(cbarticks, list):
        assert len(cbarticks) == 2
        if logscale:
            def logformatter(x, pos):
                value = np.exp(np.log(clim[0]) + x*np.log(clim[1]/clim[0]))
                value = round(value, -int(np.floor(np.log10(abs(value)))))
                return '%i' % (value)

#                values = [20.0, 50.0, 100.0, 200.0]
            pct = list(map(
                lambda x: np.log(x/clim[0])/np.log(clim[1]/clim[0]),
                cbarticks[0]
                ))

            cb.ax.yaxis.set_ticks(pct)
#                cb.ax.yaxis.set_major_formatter(
#                        mpl.ticker.FuncFormatter(logformatter))
            cb.ax.yaxis.set_major_formatter(
                    mpl.ticker.FixedFormatter(cbarticks[1])
                    )
            cb.ax.yaxis.set_tick_params(
                    labelsize=tick_fontsize,
                    direction='in',
                    pad=4.0
                    )
        else:
            cb.set_ticks(cbarticks[0])
            cb.set_ticklabels(cbarticks[1])
    else:
        if logscale:
            tick_locator = mpl.ticker.LinearLocator(numticks=cbar_nticks)
            cb.locator = tick_locator
        else:
            tick_locator = mpl.ticker.MaxNLocator(nbins=cbar_nticks)
            cb.locator = tick_locator

        cb.ax.tick_params(
            labelsize=tick_fontsize,
            direction='in',
            pad=4.0
            )
        cb.update_ticks()
