#!/usr/bin/env python
# encoding: utf-8

import os
import os.path as osp
import fnmatch

VERSION = "1.0.0"
APPNAME = "kernel-lib"

srcdir = "."
blddir = "build"


def options(opt):
    # Load modules necessary in the configuration function
    opt.load("compiler_cxx")

    # Load tools options
    opt.load("flags eigen corrade utils_cpp", tooldir="waf_tools")

    # Add options
    opt.add_option("--shared",
                   action="store_true",
                   help="build shared library")

    opt.add_option("--static",
                   action="store_true",
                   help="build static library")

    opt.add_option(
        "--multi-threading",
        action="store_true",
        help="enable multi-threading",
        dest="multi_threading",
    )


def configure(cfg):
    # OSX/Mac uses .dylib and GNU/Linux .so
    cfg.env.SUFFIX = "dylib" if cfg.env["DEST_OS"] == "darwin" else "so"

    # Load compiler configuration and generate clangd flags
    cfg.load("compiler_cxx")  # cfg.load("clang_compilation_database")

    # Define require libraries
    cfg.get_env()["requires"] += ["EIGEN", "CORRADE"]

    # Load tools configuration
    cfg.load("flags eigen corrade utils_cpp", tooldir="waf_tools")

    # Activate OPENMP if parellel option is active
    if cfg.options.multi_threading:
        cfg.load("openmp", tooldir="waf_tools")

    # Remove duplicates
    cfg.get_env()["libs"] = list(set(cfg.get_env()["libs"]))

    # Set lib type
    if cfg.options.shared:
        cfg.env["lib_type"] = "cxxshlib"
    else:
        cfg.env["lib_type"] = "cxxstlib"


def build(bld):
    # Library name
    bld.get_env()["libname"] = "Kernel"

    # Includes
    includes = []
    includes_path = "src"
    for root, _, filenames in os.walk(
            osp.join(bld.path.abspath(), includes_path)):
        for filename in fnmatch.filter(filenames, "*.hpp"):
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
            cxxxflags=bld.get_env()["CXXFLAGS"],
        )
    else:
        bld.stlib(
            features="cxx " + bld.env["lib_type"],
            source=sources,
            target=bld.get_env()["libname"],
            includes=includes_path,
            uselib=bld.get_env()["libs"],
            use="BOOST",
            cxxxflags=bld.get_env()["CXXFLAGS"],
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
