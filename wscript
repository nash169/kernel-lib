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
import os.path as osp
import fnmatch

VERSION = "1.0.0"
APPNAME = "kernel-lib"

srcdir = "."
blddir = "build"

# Tools' name and directory
tools = {"utilslib": ""}


def options(opt):
    # Load modules necessary in the configuration function
    opt.load("compiler_cxx")

    # Load personal tools options
    for key in tools:
        if tools[key]:
            opt.load(key, tooldir=os.path.join(
                tools[key], "share/waf"))

    # Load external tools options
    opt.load("flags eigen libtorch", tooldir="waf_tools")

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


def configure(cfg):
    # OSX/Mac uses .dylib and GNU/Linux .so
    cfg.env.SUFFIX = "dylib" if cfg.env["DEST_OS"] == "darwin" else "so"

    # Load compiler configuration and generate clangd flags
    try:
        # 'clang_compilation_database' required for clangd support (waf exe)
        # Waf project has to compiled with the desired tools
        # python3 ./waf-light configure build --tools=clang_compilation_database
        cfg.load("compiler_cxx clang_compilation_database")
    except:
        # Standard waf tool for C++ compilation
        cfg.load("compiler_cxx")

    # Define require libraries
    cfg.get_env()["requires"] += ["EIGEN"]

    # Load personal tools configurations
    for key in tools:
        if tools[key]:
            cfg.load(key, tooldir=os.path.join(
                tools[key], "share/waf"))

    # Load external tools configurations
    cfg.load("flags eigen libtorch", tooldir="waf_tools")

    # Activate OPENMP if parellel option is active
    if cfg.options.parallel:
        cfg.load("openmp", tooldir="waf_tools")
        cfg.env["DEFINES"] += ["PARALLEL"]

    # Remove duplicates
    cfg.get_env()["libs"] = list(set(cfg.get_env()["libs"]))

    # Set lib type
    if cfg.options.shared:
        cfg.env["lib_type"] = "cxxshlib"
    else:
        cfg.env["lib_type"] = "cxxstlib"

    # Save configuration
    cfg.env.store("build/kernellib_config.py")


def build(bld):
    # Library name
    bld.get_env()["libname"] = "Kernel"

    # Includes
    includes = []
    includes_path = "src"
    for root, _, filenames in os.walk(osp.join(bld.path.abspath(), includes_path)):
        for filename in filenames:
            if filename.endswith(('.hpp', '.h')):
                includes.append(os.path.join(root, filename))
    includes = [f[len(bld.path.abspath()) + 1:] for f in includes]

    # Sources
    sources = []
    sources_path = "src/kernel_lib"
    for root, _, filenames in os.walk(
            osp.join(bld.path.abspath(), sources_path)):
        for filename in fnmatch.filter(filenames, "*.cpp"):
            sources.append(os.path.join(root, filename))
    sources = " ".join([f[len(bld.path.abspath()) + 1:] for f in sources])

    # Build library
    if bld.options.shared:
        bld.shlib(
            features="cxx " + bld.env["lib_type"],
            source=sources,
            target=bld.get_env()["libname"],
            includes=includes_path,
            uselib=bld.get_env()["libs"],
        )
    else:
        bld.stlib(
            features="cxx " + bld.env["lib_type"],
            source=sources,
            target=bld.get_env()["libname"],
            includes=includes_path,
            uselib=bld.get_env()["libs"],
        )

    # Build executables
    bld.recurse("./src/examples")
    bld.recurse("./src/tests")
    bld.recurse("./src/benchmarks")

    # Install headers
    for f in includes:
        end_index = f.rfind("/")
        if end_index == -1:
            end_index = len(f)
        bld.install_files("${PREFIX}/include/" + f[4:end_index], f)

    # Install libraries
    if bld.env["lib_type"] == "cxxstlib":
        bld.install_files("${PREFIX}/lib",
                          blddir + "/lib" + bld.get_env()["libname"] + ".a")
    else:
        bld.install_files(
            "${PREFIX}/lib",
            blddir + "/lib" + bld.get_env()["libname"] + "." + bld.env.SUFFIX,
        )

    # Install waf tools
    bld.install_files("${PREFIX}/share/waf", "scripts/kernellib.py")

    # Install configuration file
    bld.install_files("${PREFIX}/share/waf/", "build/kernellib_config.py")
    bld.install_files("${PREFIX}/share/waf/", "waf_tools/utils.py")
