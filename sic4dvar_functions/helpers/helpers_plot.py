"""
SIC4DVAR-LC
Copyright (C) 2025 INRAE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Created in September 2023
by @Isadora Silva

Last modified on April 18th 2024 at 07:00
by @Isadora Silva

@authors: Isadora Silva
"""
import pathlib
from typing import Iterable, Tuple
import numpy as np
from matplotlib import pyplot as plt

def helper_plot_lines(xs: Iterable, ys: Iterable, show: bool=True, line_styles: Tuple=('-',), line_widths: Tuple=(1.5,), line_colors: Tuple=('black',), line_labels: Tuple=('main',), marker_styles: Tuple | None=None, marker_sizes: Tuple=(3.0,), marker_fill_colors: Tuple=('gray',), marker_edge_colors: Tuple=('gray',), title: str='', x_axis_title: str='', y_axis_title: str='', x_lim: Iterable | None=None, y_lim: Iterable | None=None, fig_width: int=5, fig_height: int=5, add_text_bottom_right: str='', add_text_bottom_left: str='', add_legend: bool=False, legend_title: str='Legend', legend_loc: str='upper center', legend_bbox_to_anchor: tuple=(1.15, 1.02), legend_ncols: int=1, output_file: str | bytes | pathlib.PurePath | None=None):
    """

    Parameters
    ----------
    xs
    ys
    show
    line_styles
    line_widths
    line_colors
    line_labels
    marker_styles: Tuple | None = None,
    marker_sizes: Tuple = (3.,),
    marker_fill_colors: Tuple = ('gray',),
    marker_edge_colors: Tuple = ('gray',),
    title
    x_axis_title
    y_axis_title
    x_lim
    y_lim
    fig_width
    fig_height
    add_text_bottom_right
    add_text_bottom_left
    add_legend
    legend_title
    legend_loc : str
        'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right',
         'lower center', 'upper center', 'center'
    legend_bbox_to_anchor
    legend_ncols
    output_file

    Returns
    -------

    """
    if output_file:
        output_file = pathlib.Path(output_file).with_suffix('.png')
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)
    if not output_file and (not show):
        raise TypeError('one of output_file or show must be specified')
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)
    if add_legend:
        if len(line_labels) != len(ys):
            raise IndexError('must define labels for all lines')
    if len(line_styles) != len(ys):
        line_styles = None
    if len(line_colors) != len(ys):
        line_colors = None
    if len(line_widths) != len(ys):
        line_widths = None
    if marker_styles is None:
        marker_sizes, marker_fill_colors, marker_edge_colors = (None, None, None)
    else:
        if len(marker_sizes) != len(ys):
            marker_sizes = None
        if len(marker_fill_colors) != len(ys):
            marker_fill_colors = None
        if len(marker_edge_colors) != len(ys):
            marker_edge_colors = marker_fill_colors
    for n, (x, y) in enumerate(zip(xs, ys)):
        plot_kwargs = {}
        if line_styles is not None:
            plot_kwargs['linestyle'] = line_styles[n]
        if line_colors is not None:
            plot_kwargs['color'] = line_colors[n]
        if line_widths is not None:
            plot_kwargs['linewidth'] = line_widths[n]
        if marker_styles is not None:
            plot_kwargs['marker'] = marker_styles[n]
        if marker_edge_colors is not None:
            plot_kwargs['markeredgecolor'] = marker_edge_colors[n]
        if marker_fill_colors is not None:
            plot_kwargs['markerfacecolor'] = marker_fill_colors[n]
        if marker_sizes is not None:
            plot_kwargs['markersize'] = marker_sizes[n]
        if add_legend:
            plot_kwargs['label'] = line_labels[n]
        ax.plot(x, y, **plot_kwargs)
    if add_legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncols=legend_ncols, title=legend_title)
    if title:
        plt.title(title)
    if x_axis_title:
        plt.xlabel(x_axis_title)
    if y_axis_title:
        plt.ylabel(y_axis_title)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid()
    if add_text_bottom_right:
        ax.text(-0.05, -0.12, add_text_bottom_right, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    if add_text_bottom_left:
        ax.text(0.7, -0.12, add_text_bottom_left, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
    if not show:
        plt.close()

def helper_plot_scatter(xs: Iterable, ys: Iterable, show: bool=True, marker_styles: Tuple='o', marker_sizes: Tuple=(5.0,), marker_fill_colors: Tuple=('gray',), marker_edge_colors: Tuple=('gray',), marker_transparency: Tuple=(10.0,), marker_labels: Tuple=('main',), title: str='', x_axis_title: str='', y_axis_title: str='', x_lim: Iterable | None=None, y_lim: Iterable | None=None, fig_width: int=5, fig_height: int=5, add_text_bottom_right: str='', add_text_bottom_left: str='', add_identity_line: bool=False, add_legend: bool=False, legend_title: str='Legend', legend_loc: str='upper center', legend_bbox_to_anchor: tuple=(1.15, 1.02), legend_ncols: int=1, output_file: str | bytes | pathlib.PurePath | None=None):
    """

    Parameters
    ----------
    xs: Iterable,
    ys: Iterable,
    show: bool
    marker_styles: Tuple
    marker_sizes: Tuple
    marker_fill_colors: Tuple
    marker_edge_colors: Tuple
    marker_labels: Tuple
    title: str
    x_axis_title: str
    y_axis_title: str
    x_lim: Iterable | None
    y_lim: Iterable | None
    fig_width: int
    fig_height: int
    add_text_bottom_right: str
    add_text_bottom_left: str
    add_identity_line: bool
    add_legend: bool
    legend_title: str
    legend_loc: str
    legend_bbox_to_anchor: tuple
    legend_ncols: int
    output_file: str | bytes | pathlib.PurePath | None
    """
    if output_file:
        output_file = pathlib.Path(output_file).with_suffix('.png')
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)
    if not output_file and (not show):
        raise TypeError('one of output_file or show must be specified')
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)
    if add_identity_line:
        x_min, x_max = np.nanquantile(np.array(xs), [0.0, 1.0])
        y_min, y_max = np.nanquantile(np.array(ys), [0.0, 1.0])
        ax.plot([x_min, x_max], [x_min, x_max], color='black')
        if x_lim is None and y_lim is None:
            x_lim = (x_min, x_max)
            y_lim = (x_min, x_max)
    if add_legend:
        if len(marker_labels) != len(ys):
            raise IndexError('must define labels for all lines')
    if len(marker_styles) != len(ys):
        marker_styles = None
    if len(marker_sizes) != len(ys):
        marker_sizes = None
    if len(marker_fill_colors) != len(ys):
        marker_fill_colors = None
    if len(marker_edge_colors) != len(ys):
        marker_edge_colors = marker_fill_colors
    if len(marker_transparency) != len(ys):
        marker_transparency = None
    for n, (x, y) in enumerate(zip(xs, ys)):
        plot_kwargs = {}
        if marker_styles is not None:
            plot_kwargs['marker'] = marker_styles[n]
        if marker_edge_colors is not None:
            plot_kwargs['edgecolors'] = marker_edge_colors[n]
        if marker_fill_colors is not None:
            plot_kwargs['c'] = marker_fill_colors[n]
        if marker_sizes is not None:
            plot_kwargs['s'] = marker_sizes[n]
        if marker_transparency is not None:
            plot_kwargs['alpha'] = (100.0 - marker_transparency[n]) / 100.0
        if add_legend:
            plot_kwargs['label'] = marker_labels[n]
        ax.scatter(x, y, **plot_kwargs)
    if add_legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncols=legend_ncols, title=legend_title)
    if title:
        plt.title(title)
    if x_axis_title:
        plt.xlabel(x_axis_title)
    if y_axis_title:
        plt.ylabel(y_axis_title)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid()
    if add_text_bottom_right:
        ax.text(-0.05, -0.12, add_text_bottom_right, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    if add_text_bottom_left:
        ax.text(0.7, -0.12, add_text_bottom_left, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
    if not show:
        plt.close()