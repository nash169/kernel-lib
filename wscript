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

import os
from wafbuild.utils import load

VERSION = "1.0.0"
APPNAME = "kernel-lib"

libname = "Kernel"
srcdir = "src"
blddir = "build"
libdir = "kernel_lib"

compiler = "cxx"
required = ["eigen"]
optional = ["libtorch", "utilslib"]


def options(opt):
    # Add options
    opt.add_option("--shared",
                   action="store_true",
                   help="build shared library")

    opt.add_option("--static",
                   action="store_true",
                   help="build static library")

    opt.add_option(
        "--parallel",
        action="store_true",
        help="enable multi-threading",
        dest="parallel",
    )

    # Load library options
    load(opt, compiler, required, optional)

    # Load examples options
    opt.recurse("./src/examples")
    opt.recurse("./src/tests")
    opt.recurse("./src/benchmarks")


def configure(cfg):
    # Load library configurations
    load(cfg, compiler, required, optional)

    # Load examples configurations
    cfg.recurse("./src/examples")
    cfg.recurse("./src/tests")
    cfg.recurse("./src/benchmarks")

    # Save configuration
    cfg.env.store("build/kernellib_config.py")


def build(bld):
    # Library name
    bld.get_env()["libname"] = libname

    # Includes
    includes = []
    for root, _, filenames in os.walk(os.path.join(srcdir, libdir)):
        includes += [os.path.join(root, filename)
                     for filename in filenames if filename.endswith(('.hpp', '.h'))]

    # Sources
    sources = []
    for root, _, filenames in os.walk(os.path.join(srcdir, libdir)):
        sources += [os.path.join(root, filename)
                    for filename in filenames if filename.endswith(('.cpp', '.cc'))]

    # Build library
    bld.shlib(
        features="cxx cxxshlib",
        source=sources,
        target=bld.get_env()["libname"],
        includes=srcdir,
        uselib=bld.get_env()["libs"],
    ) if bld.options.shared else bld.stlib(
        features="cxx cxxstlib",
        source=sources,
        target=bld.get_env()["libname"],
        includes=srcdir,
        uselib=bld.get_env()["libs"],
    )

    # Build executables
    bld.recurse("./src/examples")
    bld.recurse("./src/tests")
    bld.recurse("./src/benchmarks")

    # Install headers
    [bld.install_files("${PREFIX}/include/" + os.path.dirname(f)[4:], f)
     for f in includes]

    # Install libraries
    bld.install_files("${PREFIX}/lib", blddir + "/lib" + bld.get_env()["libname"] + "." + bld.env.SUFFIX) if bld.options.shared else bld.install_files(
        "${PREFIX}/lib", blddir + "/lib" + bld.get_env()["libname"] + ".a")

    # Install configuration file
    bld.install_files("${PREFIX}/share/waf/", "build/kernellib_config.py")
