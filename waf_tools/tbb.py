#! /usr/bin/env python
# encoding: utf-8

from waflib.Configure import conf
from utils import check_include, check_lib


def options(opt):
    opt.add_option(
        "--tbb-path", type="string", help="path to Intel TBB", dest="tbb_path"
    )


@conf
def check_tbb(ctx):
    # Set the search path
    if ctx.options.tbb_path is None:
        path_check = ["/usr/local", "/usr", "/opt/intel", "/opt/intel/tbb"]
    else:
        path_check = ctx.options.tbb_path

    # TBB includes
    check_include(ctx, "TBB", "tbb", ["parallel_for.h"], path_check)

    # TBB libs
    check_lib(ctx, "TBB", ["tbb"], path_check)

    # Add TBB
    if ctx.env.LIB_TBB:
        if not ctx.get_env()["libs"]:
            ctx.get_env()["libs"] = "TBB "
        else:
            ctx.get_env()["libs"] = ctx.get_env()["libs"] + "TBB "

        ctx.env.DEFINES_TBB = ["USE_TBB"]


def configure(cfg):
    if not cfg.env.LIB_TBB:
        cfg.check_tbb()
