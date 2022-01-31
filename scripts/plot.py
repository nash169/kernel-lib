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
from utils import get_data

data = get_data(sys.argv[1], "X", "Y", "F")

fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
surf = ax.plot_surface(
    data["X"], data["Y"], data["F"].reshape(data["X"].shape, order='F'), cmap=cm.Spectral, linewidth=0, antialiased=False
)
fig.colorbar(surf, ax=ax)
ax = fig.add_subplot(122)
contour = ax.contourf(data["X"], data["Y"], data["F"].reshape(
    data["X"].shape, order='F'), cmap=cm.Spectral)
ax.set_aspect("equal", "box")
fig.colorbar(contour, ax=ax)

# # Plot Gramian
# fig2 = plt.figure()
# plt.imshow(data["F"][:, np.newaxis])
# plt.colorbar()
# plt.show()

plt.show()
