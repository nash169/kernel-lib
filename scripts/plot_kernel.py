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

import numpy as np
import matplotlib.pyplot as plt
import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utils import draw_mesh, get_data

kernel = sys.argv[1] if len(sys.argv) > 1 else "euclidean"

if kernel == 'euclidean':
    data = get_data("outputs/kernel.csv", "X", "Y", "EVAL", "GRAM")
    X = data["X"]
    Y = data["Y"]
    F = data["EVAL"].reshape(X.shape, order='F')
    K = data["GRAM"]

    # Plot kernel
    fig_1 = plt.figure()
    ax = fig_1.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, F, cmap=cm.Spectral,
                           linewidth=0, antialiased=True)
    fig_1.colorbar(surf, ax=ax)

    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111)
    contour = ax.contourf(X, Y, F, cmap=cm.Spectral)
    ax.set_aspect("equal", "box")
    fig_2.colorbar(contour, ax=ax)

    # Plot Gramian
    fig_3 = plt.figure()
    plt.imshow(K)
    plt.colorbar()

    plt.show()

elif kernel == 'riemann':
    data = get_data("outputs/riemann.csv", "NODES",
                    "CHART", "EMBED", "MESH", "GRAM", "INDEX", "SURF")

    Xc = data["CHART"][:, 0].reshape((100, 100), order='F')
    Yc = data["CHART"][:, 1].reshape((100, 100), order='F')

    Xe = data["EMBED"][:, 0].reshape((100, 100), order='F')
    Ye = data["EMBED"][:, 1].reshape((100, 100), order='F')
    Ze = data["EMBED"][:, 2].reshape((100, 100), order='F')

    Fs = data["SURF"].reshape((100, 100), order='F')

    fig_1 = plt.figure()
    ax = fig_1.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Xc, Yc, Fs, cmap=cm.jet,
                           antialiased=True, linewidth=0)
    fig_1.colorbar(surf, ax=ax)

    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Xe, Ye, Ze, facecolors=cm.jet(
        Fs/np.amax(Fs)), antialiased=True, linewidth=0)
    fig_2.colorbar(surf, ax=ax)
    ax.set_box_aspect((np.ptp(Xe), np.ptp(Ye), np.ptp(Ze)))

    N = data["NODES"]
    I = data["INDEX"]
    F = data["MESH"]

    K = data["GRAM"]

    x, y, z = (N[:, i] for i in range(3))
    draw_mesh(x, y, z, I, F, 0)

    plt.show()
