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
    opt.load("eigen", tooldir="waf_tools")

    # Add options
    opt.add_option("--shared", action="store_true",
                   help="build shared library")
    opt.add_option("--static", action="store_true",
                   help="build static library")
    opt.add_option("--no-avx", action="store_true",
                   help="build without AVX flags", dest="disable_avx",)


def configure(cfg):
    # Load modules defined in the option function
    cfg.load("compiler_cxx")
    cfg.load("compiler_c")
    cfg.load("eigen")
    cfg.load("tbb")
    cfg.load("mkl")
    cfg.load("avx")

    # Don't know... define some kind of lib type (check this)
    cfg.env["lib_type"] = "cxxstlib"
    if cfg.options.shared:
        cfg.env["lib_type"] = "cxxshlib"

    # Magical flags definition for different compilers from Konst... hard to say what eachone does
    if cfg.env.CXX_NAME in ["icc", "icpc"]:
        common_flags = "-Wall -std=c++14"
        opt_flags = " -O3 -xHost -mtune=native -unroll -g"
    elif cfg.env.CXX_NAME in ["clang"]:
        common_flags = "-Wall -std=c++14"
        opt_flags = " -O3 -march=native -g -faligned-new"
    else:
        gcc_version = int(cfg.env["CC_VERSION"][0] + cfg.env["CC_VERSION"][1])
        if gcc_version < 47:
            common_flags = "-Wall -std=c++0x"
            # cfg.fatal("Compiler should support C++14")
        else:
            common_flags = "-Wall -std=c++14"
        opt_flags = " -O3 -march=native -g"
        if gcc_version >= 71:
            opt_flags = opt_flags + " -faligned-new"

    all_flags = common_flags + opt_flags
    cfg.env["CXXFLAGS"] = cfg.env["CXXFLAGS"] + all_flags.split(" ")
    print(cfg.env["CXXFLAGS"])


def build(bld):
    # OSX/Mac uses .dylib and GNU/Linux .so
    suffix = "dylib" if bld.env["DEST_OS"] == "darwin" else "so"

    # Define necessary libraries
    libs = "BOOST EIGEN "

    # Get flags
    cxxflags = bld.get_env()["CXXFLAGS"]

    # Define needed libraries
    libname = "kernelLib"
    bld.get_env()["kernel_lib_libname"] = libname
    bld.get_env()["kernel_lib_libs"] = libs

    # Check if all the libraries have been loaded correctly
    if len(bld.env.INCLUDES_EIGEN) == 0:
        bld.fatal("Some libraries were not found! Cannot proceed!")

    # Define sources files
    files = []
    for root, dirnames, filenames in os.walk(bld.path.abspath() + "/src/kernel_lib/"):
        for filename in fnmatch.filter(filenames, "*.cpp"):
            files.append(os.path.join(root, filename))

    files = [f[len(bld.path.abspath()) + 1:] for f in files]
    kernel_lib_srcs = " ".join(files)

    # Build library
    if bld.options.shared:
        bld.shlib(
            features="cxx " + bld.env["lib_type"],
            source=kernel_lib_srcs,
            target=libname,
            includes="./src",
            uselib=libs,
            cxxxflags=cxxflags,
        )
    else:
        bld.stlib(
            features="cxx " + bld.env["lib_type"],
            source=kernel_lib_srcs,
            target=libname,
            includes="./src",
            uselib=libs,
            cxxxflags=cxxflags,
        )

    bld.recurse("./src/examples")

    # Define headers to install
    install_files = []
    for root, dirnames, filenames in os.walk(bld.path.abspath() + "/src/"):
        for filename in fnmatch.filter(filenames, "*.hpp"):
            install_files.append(os.path.join(root, filename))
    install_files = [f[len(bld.path.abspath()) + 1:] for f in install_files]

    # Install headers
    for f in install_files:
        end_index = f.rfind("/")
        if end_index == -1:
            end_index = len(f)
        bld.install_files("${PREFIX}/include/" + f[4:end_index], f)

    # Install libraries
    if bld.env["lib_type"] == "cxxstlib":
        bld.install_files("${PREFIX}/lib", blddir + "/kernel_lib.a")
    else:
        bld.install_files("${PREFIX}/lib", blddir + "/kernel_lib." + suffix)
