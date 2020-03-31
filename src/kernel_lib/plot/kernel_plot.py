#! /usr/bin/env python
# encoding: utf-8

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def get_line(name, fs):
    while True:
        line = next(fs)
        if name in line:
            next(fs)
            while True:
                line = next(fs)
                if line in ["\n", "\r\n"]:
                    break
                else:
                    yield line
        elif not line:
            break


def get_data(file_path, var):
    with open(file_path) as fs:
        g = get_line(var, fs)
        M = np.loadtxt(g)

    return M


rsc_path = os.path.join(os.getcwd(), "../../../rsc")
file_path = os.path.join(rsc_path, "eval_data.csv")

X = get_data(file_path, "X")
Y = get_data(file_path, "Y")
F = get_data(file_path, "F")

fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
surf = ax.plot_surface(X, Y, F, cmap=cm.Spectral, linewidth=0, antialiased=False)
fig.colorbar(surf, ax=ax)
ax = fig.add_subplot(122)
contour = ax.contourf(X, Y, F, cmap=cm.Spectral)
ax.set_aspect("equal", "box")
fig.colorbar(contour, ax=ax)
