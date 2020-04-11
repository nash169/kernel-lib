#! /usr/bin/env python
# encoding: utf-8

import numpy as np


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


def get_data(file_path, *args):
    M = {}
    for var in args:
        with open(file_path) as fs:
            g = get_line(var, fs)
            M[var] = np.loadtxt(g)

    return M
