#! /usr/bin/env python
# encoding: utf-8

from waflib.Configure import conf


def options(opt):
    opt.add_option(
        "--debug-flags",
        action="store_true",
        help="activate debug flags",
        dest="debug_flags",
    )
    opt.add_option(
        "--release-flags",
        action="store_true",
        help="activate release flags",
        dest="release_flags",
    )
    opt.add_option(
        "--optional-flags",
        action="store_true",
        help="activate optional flags",
        dest="optional_flags",
    )


@conf
def check_flags(ctx):
    ctx.env["CXXFLAGS"] = ["-Wall", "-w"]

    if ctx.env.CXX_NAME in ["icc", "icpc"]:
        ctx.env["CXXFLAGS"].append("-std=c++11")
        opt_flags = ["-O3", "-xHost", "-mtune=native", "-unroll", "-g"]
    elif ctx.env.CXX_NAME in ["clang"]:
        ctx.env["CXXFLAGS"].append("-std=c++11")
        opt_flags = ["-O3", "-march=native", "-g", "-faligned-new"]
    elif ctx.env.CXX_NAME in ["gcc", "g++"]:
        gcc_v = int(ctx.env["CC_VERSION"][0] + ctx.env["CC_VERSION"][1])
        ctx.env["CXXFLAGS"].append("-std=c++11" if gcc_v >= 47 else "-std=c++0x")
        opt_flags = [
            "-O3",
            "-march=native",
            "-g",
            ("-faligned-new" if gcc_v >= 47 else None),
        ]

    if ctx.options.optional_flags:
        ctx.env["CXXFLAGS"] = ctx.env["CXXFLAGS"] + opt_flags


def configure(cfg):
    cfg.check_flags()
