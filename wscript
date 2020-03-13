#!/usr/bin/env python
# encoding: utf-8

import os
import fnmatch

VERSION = "1.0.0"
APPNAME = "kernel-lib"

srcdir = "."
blddir = "build"


def options(opt):
    # Load modules necessary in the configuration function
    opt.load("compiler_cxx")
    opt.load("compiler_c")

    # Load flags options
    opt.load("flags", tooldir="waf_tools")

    # Load tools options
    opt.load("eigen", tooldir="waf_tools")

    # Add options
    opt.add_option("--shared", action="store_true",
                   help="build shared library")
    opt.add_option("--static", action="store_true",
                   help="build static library")
    opt.add_option("--no-avx", action="store_true",
                   help="build without AVX flags", dest="disable_avx",)


def configure(cfg):
    # OSX/Mac uses .dylib and GNU/Linux .so
    cfg.env.SUFFIX = "dylib" if cfg.env["DEST_OS"] == "darwin" else "so"

    # Load compiler configuration
    cfg.load("compiler_cxx")
    cfg.load("compiler_c")

    # Load compiler flags
    cfg.load("flags", tooldir="waf_tools")

    # Load tools configuration
    cfg.load("eigen", tooldir="waf_tools")

    # Set lib type
    if cfg.options.static:
        cfg.env["lib_type"] = "cxxstlib"
    else:
        cfg.env["lib_type"] = "cxxshlib"


def build(bld):
    # Library name
    bld.get_env()["libname"] = "kernelLib"

    # Includes
    includes = []
    includes_path = "src"
    for root, dirnames, filenames in os.walk(bld.path.abspath() + includes_path):
        for filename in fnmatch.filter(filenames, "*.hpp"):
            includes.append(os.path.join(root, filename))
    includes = [f[len(bld.path.abspath()) + 1:] for f in includes]

    # Sources
    sources = []
    sources_path = "src/kernel_lib"
    for root, dirnames, filenames in os.walk(bld.path.abspath() + sources_path):
        for filename in fnmatch.filter(filenames, "*.cpp"):
            sources.append(os.path.join(root, filename))
    sources = " ".join([f[len(bld.path.abspath()) + 1:] for f in sources])

    # Build library
    if bld.options.static:
        bld.stlib(
            features="cxx " + bld.env["lib_type"],
            source=sources,
            target=bld.get_env()["libname"],
            includes=includes_path,
            uselib=bld.get_env()["libs"],
            cxxxflags=bld.get_env()["CXXFLAGS"],
        )
    else:
        bld.shlib(
            features="cxx " + bld.env["lib_type"],
            source=sources,
            target=bld.get_env()["libname"],
            includes=includes_path,
            uselib=bld.get_env()["libs"],
            cxxxflags=bld.get_env()["CXXFLAGS"],
        )

    # Build examples
    bld.recurse("./src/examples")

    # Install headers
    for f in includes:
        end_index = f.rfind("/")
        if end_index == -1:
            end_index = len(f)
        bld.install_files("${PREFIX}/include/" + f[4:end_index], f)

    # Install libraries
    if bld.env["lib_type"] == "cxxstlib":
        bld.install_files(
            "${PREFIX}/lib", blddir + "/lib" + bld.get_env()["libname"] + ".a"
        )
    else:
        bld.install_files(
            "${PREFIX}/lib",
            blddir + "/lib" + bld.get_env()["libname"] + "." + bld.env.SUFFIX,
        )
