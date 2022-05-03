#!/usr/bin/env python
# encoding: utf-8
#
#    This file is part of kernel-lib.
#
#    Copyright (c) 2020, 2021, 2022 Bernardo Fichera <bernardo.fichera@gmail.com>
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

from turtle import color
import numpy as np


def get_line(name, fs):
    while True:
        try:
            line = next(fs)
        except StopIteration:
            return

        if name in line:
            try:
                next(fs)
            except StopIteration:
                return

            while True:
                try:
                    line = next(fs)
                except StopIteration:
                    return

                if line in ["\n", "\r\n"]:
                    break
                else:
                    yield line
        elif not line:
            break


def get_data(file_path, *args):
    M = {}
    for var in args:
        with open(file_path) as fs:
            g = get_line(var, fs)
            M[var] = np.loadtxt(g)

    return M


def draw_mesh(x, y, z, triangles, function, center=0):
    from mayavi import mlab
    mlab.triangular_mesh(x, y, z, triangles, scalars=function)
    v_options = {'mode': 'sphere',
                 'scale_factor': 1e-1, }
    mlab.points3d(x[center], y[center], z[center], **v_options)


def draw_field(x, y, z, u, v, w, triangles, function, center=0):
    from mayavi import mlab
    mlab.triangular_mesh(x, y, z, triangles, color=(1, 1, 1))
    v_options = {'mode': 'sphere',
                 'scale_factor': 1e-1}
    mlab.points3d(x[center], y[center], z[center], **v_options)
    obj = mlab.quiver3d(x, y, z, u, v, w, scalars=function, scale_mode='none')
    obj.glyph.color_mode = 'color_by_scalar'
