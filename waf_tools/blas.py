#! /usr/bin/env python
# encoding: utf-8

from waflib.Configure import conf
from utils import check_lib


def options(opt):
    opt.add_option(
        "--blas-path", type="string", help="path to OpenBLAS", dest="blas_path"
    )


@conf
def check_blas(ctx):
    # Set the search path
    if ctx.options.blas_path is None:
        path_check = ["/usr/local", "/usr", "/opt"]
    else:
        path_check = ctx.options.blas_path

    # LIB Check
    check_lib(ctx, "BLAS", ["blas"], path_check)

    if ctx.env.LIB_BLAS:
        if not ctx.get_env()["libs"]:
            ctx.get_env()["libs"] = "BLAS "
        else:
            ctx.get_env()["libs"] = ctx.get_env()["libs"] + "BLAS "


def configure(cfg):
    if not cfg.env.LIB_BLAS:
        cfg.check_blas()
