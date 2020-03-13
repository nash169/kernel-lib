#! /usr/bin/env python
# encoding: utf-8

from waflib.Configure import conf
from utils import check_lib


def options(opt):
    opt.add_option(
        "--lapack-path", type="string", help="path to LAPACK", dest="lapack_path"
    )


@conf
def check_lapack(ctx):
    # Set the search path
    if ctx.options.lapack_path is None:
        path_check = ["/usr/local", "/usr", "/opt"]
    else:
        path_check = ctx.options.lapack_path

    # Check LAPACK libs
    check_lib(ctx, "LAPACK", ["lapacke"], path_check)

    # Add LAPACK
    if ctx.env.LIB_LAPACK:
        if not ctx.get_env()["libs"]:
            ctx.get_env()["libs"] = "LAPACK "
        else:
            ctx.get_env()["libs"] = ctx.get_env()["libs"] + "LAPACK "


def configure(cfg):
    if not cfg.env.LIB_LAPACK:
        cfg.check_lapack()
