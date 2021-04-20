#! /usr/bin/env python
# encoding: utf-8

from waflib.Configure import conf
from utils import check_include, check_lib


def options(opt):
    # Required package options
    opt.load("eigen corrade", tooldir="waf_tools")

    # Options
    opt.add_option(
        "--kernel-path", type="string", help="path to kernel-lib", dest="kernel_path"
    )


@conf
def check_kernel(ctx):
    # Set the search path
    if ctx.options.kernel_path is None:
        path_check = ["/usr/local", "/usr"]
    else:
        path_check = [ctx.options.kernel_path]

    # kernel-lib includes
    check_include(ctx, "KERNEL", ["kernel_lib"], ["Kernel.hpp"], path_check)

    # kernel-lib libs
    check_lib(ctx, "KERNEL", "", ["libKernel"], path_check)

    if ctx.env.LIB_KERNEL or ctx.env.STLIB_KERNEL:
        # Add dependencies to require libraries
        ctx.get_env()["requires"] = ctx.get_env()[
            "requires"] + ["EIGEN", "CORRADE"]

        # Check for dependencies
        ctx.load("eigen corrade", tooldir="waf_tools")

        # Add library
        ctx.get_env()["libs"] = ctx.get_env()["libs"] + ["KERNEL"]


def configure(cfg):
    if not cfg.env.LIB_KERNEL and not cfg.env.STLIB_KERNEL:
        cfg.check_kernel()
