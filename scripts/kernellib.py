#!/usr/bin/env python
# encoding: utf-8
#
#    This file is part of kernel-lib.
#
#    Copyright (c) 2020, 2021, 2022 Bernardo Fichera <bernardo.fichera@gmail.com>
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

from waflib.Configure import conf
from utils import check_include, check_lib, check_config


def options(opt):
    # Required package options
    opt.load("eigen corrade", tooldir="waf_tools")

    # Options
    opt.add_option(
        "--kernellib-path", type="string", help="path to kernel-lib", dest="kernellib_path"
    )
    # Load configuration file
    opt.add_option(
        "--kernellib-config", action="store_true", help="load kernel-lib configuration", dest="kernellib_config"
    )


@conf
def check_kernellib(ctx):
    # Set the search path
    if ctx.options.kernellib_path is None:
        path_check = ["/usr/local", "/usr"]
    else:
        path_check = [ctx.options.kernellib_path]

    # kernel-lib includes
    check_include(ctx, "KERNELLIB", [""], [
                  "kernel_lib/Kernel.hpp"], path_check)

    # kernel-lib libs
    check_lib(ctx, "KERNELLIB", "", ["libKernel"], path_check)

    # Check configuration
    if ctx.options.kernellib_config:
        check_config(ctx, "", "kernellib_config.py", path_check)

    if ctx.env.LIB_KERNELLIB or ctx.env.STLIB_KERNELLIB:
        # Add dependencies to require libraries
        ctx.get_env()["requires"] = ctx.get_env()[
            "requires"] + ["EIGEN", "CORRADE"]

        # Check for dependencies
        ctx.load("eigen corrade", tooldir="waf_tools")

        # Add library
        ctx.get_env()["libs"] += ["KERNELLIB"]


def configure(cfg):
    if not cfg.env.LIB_KERNELLIB and not cfg.env.STLIB_KERNELLIB:
        cfg.check_kernellib()
