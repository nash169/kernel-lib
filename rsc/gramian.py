#! /usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utils import get_data

data = get_data(sys.argv[1], "gramian")

fig = plt.figure()
plt.imshow(data["gramian"])
plt.colorbar()
plt.show()
