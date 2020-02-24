#! /usr/bin/env python
# encoding: utf-8

"""
Quick n dirty intel mkl detection
"""

import os, glob, types
from waflib.Configure import conf
import kernel_lib


def options(opt):
    opt.add_option(
        "--mkl", type="string", help="path to Intel Math Kernel Library", dest="mkl"
    )


@conf
def check_mkl(conf):
    if conf.options.mkl:
        includes_mkl = [conf.options.mkl + "/include"]
        libpath_mkl = [conf.options.mkl + "/lib/intel64", conf.options.mkl + "/lib/"]
    else:
        includes_mkl = ["/opt/intel/mkl/include", "/usr/local/include", "/usr/include"]
        libpath_mkl = [
            "/opt/intel/mkl/lib",
            "/usr/local/lib/",
            "/usr/lib",
            "/opt/intel/mkl/lib/intel64",
            "/usr/lib/x86_64-linux-gnu/",
        ]

    conf.start_msg("Checking Intel MKL includes (optional)")
    try:
        res = conf.find_file("mkl.h", includes_mkl)
        conf.end_msg("ok")
        conf.start_msg("Checking Intel MKL libs (optional)")
        kernel_lib.check_lib(conf, "libmkl_core", libpath_mkl)
        conf.end_msg("ok")
    except:
        conf.end_msg("Not found", "RED")
        return
    conf.env.LIB_MKL_SEQ = [
        "mkl_intel_lp64",
        "mkl_core",
        "mkl_sequential",
        "pthread",
        "m",
    ]
    conf.env.LIB_MKL_TBB = [
        "mkl_intel_lp64",
        "mkl_core",
        "mkl_tbb_thread",
        "tbb",
        "stdc++",
        "pthread",
        "m",
    ]
    if conf.env.CXX_NAME in ["icc", "icpc"]:
        conf.env.LIB_MKL_OMP = [
            "mkl_intel_lp64",
            "mkl_core",
            "mkl_intel_thread",
            "pthread",
            "m",
        ]
    else:
        conf.env.LIB_MKL_OMP = [
            "mkl_intel_lp64",
            "mkl_core",
            "mkl_gnu_thread",
            "dl",
            "pthread",
            "m",
        ]
    conf.env.INCLUDES_MKL_SEQ = includes_mkl
    conf.env.INCLUDES_MKL_TBB = includes_mkl
    conf.env.INCLUDES_MKL_OMP = includes_mkl
    conf.env.LIBPATH_MKL_SEQ = libpath_mkl
    conf.env.LIBPATH_MKL_TBB = libpath_mkl
    conf.env.LIBPATH_MKL_OMP = libpath_mkl
    conf.env.CXXFLAGS_MKL_SEQ = [
        "-m64",
        "-DEIGEN_USE_MKL_ALL",
        "-DMKL_BLAS=MKL_DOMAIN_BLAS",
    ]
    # conf.env.LINKFLAGS_MKL_SEQ = [ "-Wl,--no-as-needed" ]
    conf.env.CXXFLAGS_MKL_TBB = [
        "-m64",
        "-DEIGEN_USE_MKL_ALL",
        "-DMKL_BLAS=MKL_DOMAIN_BLAS",
    ]
    # conf.env.LINKFLAGS_MKL_TBB = [ "-Wl,--no-as-needed" ]
    if conf.env.CXX_NAME in ["icc", "icpc"]:
        conf.env.CXXFLAGS_MKL_OMP = [
            "-qopenmp",
            "-m64",
            "-DEIGEN_USE_MKL_ALL",
            "-DMKL_BLAS=MKL_DOMAIN_BLAS",
        ]
    else:
        conf.env.CXXFLAGS_MKL_OMP = [
            "-fopenmp",
            "-m64",
            "-DEIGEN_USE_MKL_ALL",
            "-DMKL_BLAS=MKL_DOMAIN_BLAS",
        ]
    # conf.env.LINKFLAGS_MKL_OMP = [ "-Wl,--no-as-needed" ]

