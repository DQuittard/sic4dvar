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

Description:
    Functions for saving data in gnuplot format.
"""
from copy import deepcopy
import numpy as np

def gnuplot_save(distance, times, elevation, width, location, zmin, spaces):
    all_data = []
    zmin2 = np.nanmin(elevation)
    file = open(location, 'w')
    for i in range(elevation.shape[1]):
        x = distance
        z = elevation[:, i]
        t = np.repeat(times[i], elevation[:, i].shape[0])
        w = width[:, i]
        for k in range(elevation.shape[0]):
            z[k] = z[k]
        j = 0
        for x1, z1, t1, w1 in zip(x, z, t, w):
            file.write('{} {} {} {}\n'.format(t1, x1, z1, w1))
            j += 1
            if j == elevation.shape[0]:
                if spaces <= 1:
                    file.write('\n')
                if spaces >= 2:
                    file.write('\n\n')
    file.close()

def gnuplot_save_list(distance, times, elevation, width, location, zmin, spaces):
    all_data = []
    file = open(location, 'w')
    max_length = max((len(arr) for arr in elevation))
    elevation2 = deepcopy(elevation)
    width2 = deepcopy(width)
    for i in range(0, len(elevation)):
        elevation2[i] = np.pad(elevation2[i], (0, max_length - len(elevation2[i])))
        width2[i] = np.pad(width2[i], (0, max_length - len(width2[i])))
    elevation2 = np.array(elevation2)
    width2 = np.array(width2)
    times = np.arange(max_length)
    zmin2 = np.nanmin(elevation2)
    for i in range(0, elevation2.shape[1]):
        x = distance
        z = elevation2[:, i]
        t = np.repeat(times[i], elevation2[:, i].shape[0])
        w = width2[:, i]
        j = 0
        for x1, z1, t1, w1 in zip(x, z, t, w):
            file.write('{} {} {} {}\n'.format(t1, x1, z1, w1))
            j += 1
            if j == elevation2.shape[0]:
                if spaces <= 1:
                    file.write('\n')
                if spaces >= 2:
                    file.write('\n\n')
    file.close()

def gnuplot_save_zm(zm_array_save, times, location, spaces):
    all_data = []
    file = open(location, 'w')
    for zb_index in range(0, len(zm_array_save)):
        x = np.arange(0, len(zm_array_save[zb_index]))
        z = zm_array_save[zb_index]
        t = np.repeat(zb_index, len(zm_array_save[zb_index]))
        j = 0
        for x1, z1, t1 in zip(x, z, t):
            file.write('{} {} {} \n'.format(t1, x1, z1))
            j += 1
            if j == len(zm_array_save[zb_index]):
                if spaces <= 1:
                    file.write('\n')
                if spaces >= 2:
                    file.write('\n\n')
    file.close()

def gnuplot_save_q_pdf(values, location):
    all_data = []
    file = open(location, 'w')
    for i in range(0, values.shape[0]):
        for j in range(0, values[0].shape[0]):
            file.write('{} {} {} \n'.format(values[i, j], i, j))
        file.write('\n')
    file.close()

def gnuplot_save_q_station(discharge, dates, location):
    all_data = []
    file = open(location, 'w')
    x = discharge
    t = dates
    j = 0
    for x1, t1 in zip(x, t):
        file.write('{} {}\n'.format(x1, t1))
        j += 1
    file.close()

def gnuplot_save_q(discharge, dates, location):
    all_data = []
    file = open(location, 'w')
    x = discharge
    t = dates
    j = 0
    for x1, t1 in zip(x, t):
        file.write('{} {}\n'.format(x1, t1))
        j += 1
    file.close()

def gnuplot_save_cs(width_cs, wse_cs, nodes, times, location, spaces=2):
    all_data = []
    file = open(location, 'w')
    for i in range(0, len(times)):
        w = width_cs[i]
        z = wse_cs[i]
        t = np.repeat(times[i], len(width_cs[i]))
        x = nodes
        j = 0
        for x1, w1, z1 in zip(x, w, z):
            file.write('{} {} {}\n'.format(x1, w1, z1))
            j += 1
            if j == len(t):
                if spaces <= 1:
                    file.write('\n')
                if spaces >= 2:
                    file.write('\n\n')
    file.close()

def gnuplot_save_q2(discharge, nodes, times, location, spaces=2):
    all_data = []
    file = open(location, 'w')
    for i in range(0, len(times)):
        q = discharge[i]
        t = np.repeat(times[i], len(discharge[i]))
        x = nodes
        j = 0
        for x1, q1, t1 in zip(x, q, t):
            file.write('{} {} {}\n'.format(t1, x1, q1))
            j += 1
            if j == len(t):
                if spaces <= 1:
                    file.write('\n')
                if spaces >= 2:
                    file.write('\n\n')
    file.close()

def gnuplot_save_slope(slope, times, location):
    all_data = []
    file = open(location, 'w')
    x = slope
    t = np.around(times)
    t = t - min(t)
    j = 0
    for x1, t1 in zip(x, t):
        file.write('{} {}\n'.format(x1, t1))
        j += 1
    file.close()

def gnuplot_save_c1c2(node_x, c1, c2, times, location, spaces=2):
    all_data = []
    file = open(location, 'w')
    for i in range(len(times)):
        x = node_x
        z = c1 * node_x + c2
        t = np.around(times)
        t = t - min(t)
        t = np.repeat(t[i], node_x.shape[0])
        j = 0
        for x1, z1, t1 in zip(x, z, t):
            file.write('{} {} {} \n'.format(t1, x1, z1))
            j += 1
            if j == node_x.shape[0]:
                if spaces <= 1:
                    file.write('\n')
                if spaces >= 2:
                    file.write('\n\n')
    file.close()

def gnuplot_save_tables(x, y, z, location, spaces):
    all_data = []
    file = open(location, 'w')
    j = 0
    for x1, y1, z1 in zip(x, y, z):
        file.write('{} {} {}\n'.format(x1, y1, z1))
        j += 1
        if j == len(x):
            if spaces <= 1:
                file.write('\n')
            if spaces >= 2:
                file.write('\n\n')
    file.close()

def gnuplot_save_var(x, y, z, a, b, location, spaces):
    all_data = []
    file = open(location, 'w')
    j = 0
    for x1, y1, z1, a1, b1 in zip(x, y, z, a, b):
        file.write('{} {} {} {} {}\n'.format(x1, y1, z1, a1, b1))
        j += 1
        if j == len(x):
            if spaces <= 1:
                file.write('\n')
            if spaces >= 2:
                file.write('\n\n')
    file.close()