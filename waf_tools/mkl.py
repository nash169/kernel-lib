#! /usr/bin/env python
# encoding: utf-8

from waflib.Configure import conf
from utils import check_include, check_lib


def options(opt):
    opt.add_option(
        "--mkl-path",
        type="string",
        help="path to Intel Math Kernel Library",
        dest="mkl_path",
    )
    opt.add_option(
        "--mkl-threading",
        type="string",
        help="mkl threading layer",
        dest="mkl_threading",
    )

    opt.load("tbb", tooldir="waf_tools")


@conf
def check_mkl(ctx):
    # Set the search path
    if ctx.options.mkl_path is None:
        path_check = ["/usr/local", "/usr", "/opt/intel", "/opt/intel/mkl"]
    else:
        path_check = [ctx.options.mkl_path]

    # MKL includes
    check_include(ctx, "MKL", "", ["mkl.h"], path_check)

    # MKL libs
    if ctx.options.mkl_threading is None or ctx.options.mkl_threading == "sequential":
        check_lib(
            ctx,
            "MKL",
            ["mkl_intel_ilp64", "mkl_sequential", "mkl_core"],
            path_check,
        )
        # It would be necessary to check for the presence of pthread, m and dl
        ctx.env.LIB_MKL = ctx.env.LIB_MKL + ["pthread", "m", "dl"]
    elif ctx.options.mkl_threading == "openmp":
        check_lib(
            ctx,
            "MKL",
            [
                "mkl_intel_ilp64",
                "mkl_intel_thread",
                "mkl_core",
            ],
            path_check,
        )
        # Here it would be necessary to check for the presence of OPENMP and the others above
        ctx.env.LIB_MKL = ctx.env.LIB_MKL + \
            "iomp5" if ctx.env.CXXNAME in ["icc", "icpc"] else "gomp"
        # It would be necessary to check for the presence of pthread, m and dl
        ctx.env.LIB_MKL = ctx.env.LIB_MKL + ["pthread", "m", "dl"]
    elif ctx.options.mkl_threading == "tbb":
        ctx.get_env()["requires"] = ctx.get_env()["requires"] + ["TBB"]
        ctx.load("tbb", tooldir="waf_tools")
        check_lib(
            ctx,
            "MKL",
            ["mkl_intel_ilp64", "mkl_tbb_thread",
                "mkl_core"],
            path_check,
        )
        # stdc++ has to be checked? Does it work on macOS?
        ctx.env.LIB_MKL = ctx.env.LIB_MKL + ["stdc++", "pthread", "m", "dl"]
    else:
        pass

    # Add MKL
    if ctx.env.LIB_MKL:
        # MKL flags
        if ctx.env.CXX_NAME in ["gcc", "g++"]:
            ctx.env["CXXFLAGS"] = ctx.env["CXXFLAGS"] + \
                ["-Wl,--no-as-needed", "-m64"]
        # MKL defines
        ctx.env.DEFINES_MKL = ["MKL_ILP64"]

        if not ctx.get_env()["libs"]:
            ctx.get_env()["libs"] = "MKL "
        else:
            ctx.get_env()["libs"] = ctx.get_env()["libs"] + "MKL "


def configure(cfg):
    if not cfg.env.LIB_MKL:
        cfg.check_mkl()
