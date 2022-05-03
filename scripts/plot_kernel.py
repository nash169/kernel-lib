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

import sys

import numpy as np
from scipy.linalg import norm

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import pyvista as pv


from utils import draw_field, draw_mesh, get_data

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
    res = 100
    data = get_data("outputs/riemann.csv", "NODES",
                    "CHART", "EMBED", "MESH", "GRAM", "INDEX", "SURF", "GRADIENT", "PROJ")

    Xc = data["CHART"][:, 0].reshape((res, res), order='F')
    Yc = data["CHART"][:, 1].reshape((res, res), order='F')

    Xe = data["EMBED"][:, 0].reshape((res, res), order='F')
    Ye = data["EMBED"][:, 1].reshape((res, res), order='F')
    Ze = data["EMBED"][:, 2].reshape((res, res), order='F')

    Fs = data["SURF"].reshape((res, res), order='F')
    Fs -= np.min(Fs)
    Fs /= np.max(Fs)
    data["GRADIENT"] = np.divide(data["GRADIENT"], norm(
        data["GRADIENT"], axis=1)[:, np.newaxis])
    GsX = data["GRADIENT"][:, 0]  # .reshape((res, res), order='F')
    GsY = data["GRADIENT"][:, 1]  # .reshape((res, res), order='F')
    GsZ = data["GRADIENT"][:, 2]  # .reshape((res, res), order='F')
    GsXp = data["PROJ"][:, 0]  # .reshape((res, res), order='F')
    GsYp = data["PROJ"][:, 1]  # .reshape((res, res), order='F')
    GsZp = data["PROJ"][:, 2]  # .reshape((res, res), order='F')

    # Chart surface representation
    fig_1 = plt.figure()
    ax = fig_1.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Xc, Yc, Fs, cmap=cm.jet,
                           antialiased=True, linewidth=0)
    fig_1.colorbar(surf, ax=ax)

    # Embedding surface representation
    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Xe, Ye, Ze, facecolors=cm.jet(
        Fs), antialiased=True, linewidth=0)
    fig_2.colorbar(surf, ax=ax)
    ax.set_box_aspect((np.ptp(Xe), np.ptp(Ye), np.ptp(Ze)))

    # # Non-projected vector field
    # fig_3 = plt.figure()
    # ax = fig_3.add_subplot(111, projection="3d")
    # skip = 5
    # ax.quiver(Xe[::skip, ::skip], Ye[::skip, ::skip], Ze[::skip, ::skip],
    #           GsX[::skip, ::skip], GsY[::skip, ::skip], GsZ[::skip, ::skip], length=0.1)
    # ax.set_box_aspect((np.ptp(Xe), np.ptp(Ye), np.ptp(Ze)))

    # # Projected vector field
    # fig_4 = plt.figure()
    # ax = fig_4.add_subplot(111, projection="3d")
    # surf = ax.plot_surface(Xe, Ye, Ze, color='g',
    #                        antialiased=True, linewidth=0)
    # ax.quiver(Xe.flatten()[::skip], Ye.flatten()[::skip], Ze.flatten()[::skip],
    #           GsXp.flatten()[::skip], GsYp.flatten()[::skip], GsZp.flatten()[::skip], length=0.1, color=cm.jet(
    #     Fs.flatten()))
    # ax.set_box_aspect((np.ptp(Xe), np.ptp(Ye), np.ptp(Ze)))

    # plt.show()

    # Mesh kernel solution
    N = data["NODES"]
    I = data["INDEX"]
    F = data["MESH"]

    K = data["GRAM"]

    x, y, z = (N[:, i] for i in range(3))
    # draw_mesh(x, y, z, I, F, 0)
    draw_field(x, y, z, GsXp, GsYp, GsZp, I, F, 0)
