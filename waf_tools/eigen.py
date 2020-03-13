#! /usr/bin/env python
# encoding: utf-8

from waflib.Configure import conf
from utils import check_include


def options(opt):
    opt.add_option(
        "--eigen-path", type="string", help="path to eigen", dest="eigen_path"
    )
    opt.add_option(
        "--with-lapack", action="store_true", help="enable LAPACK", dest="eigen_lapack"
    )
    opt.add_option(
        "--with-blas", action="store_true", help="enable OpenBLAS", dest="eigen_blas"
    )
    opt.add_option(
        "--with-mkl", action="store_true", help="enable MKL", dest="eigen_mkl"
    )
    opt.add_option(
        "--multi-threading",
        action="store_true",
        help="enable OpenMP",
        dest="eigen_openmp",
    )

    opt.load("lapack", tooldir="waf_tools")
    opt.load("blas", tooldir="waf_tools")
    opt.load("mkl", tooldir="waf_tools")


@conf
def check_eigen(ctx):
    # Set the search path
    if ctx.options.eigen_path is None:
        path_check = ["/usr/local", "/usr", "/opt"]
    else:
        path_check = ctx.options.eigen_path

    # EIGEN includes
    check_include(ctx, "EIGEN", "eigen3", ["Eigen/Core"], path_check)

    # Add EIGEN
    if ctx.env.INCLUDES_EIGEN:
        if not ctx.get_env()["libs"]:
            ctx.get_env()["libs"] = "EIGEN "
        else:
            ctx.get_env()["libs"] = ctx.get_env()["libs"] + "EIGEN "

        if ctx.options.eigen_lapack:
            ctx.get_env()["requires"] = ctx.get_env()["requires"] + ["LAPACK"]
            ctx.load("lapack", tooldir="waf_tools")
            ctx.env.DEFINES_EIGEN.append("EIGEN_USE_LAPACKE")

        if ctx.options.eigen_blas:
            ctx.get_env()["requires"] = ctx.get_env()["requires"] + ["BLAS"]
            ctx.load("blas", tooldir="waf_tools")
            ctx.env.DEFINES_EIGEN.append("EIGEN_USE_BLAS")

        if ctx.options.eigen_mkl:
            ctx.get_env()["requires"] = ctx.get_env()["requires"] + ["MKL"]
            ctx.load("mkl", tooldir="waf_tools")
            ctx.env.DEFINES_EIGEN.append("EIGEN_USE_MKL_VML")

        if ctx.options.eigen_openmp and ctx.options.eigen_mkl is None:
            ctx.load("openmp", tooldir="waf_tools")


def configure(cfg):
    if not cfg.env.INCLUDES_EIGEN:
        cfg.check_eigen()
