#! /usr/bin/env python
# encoding: utf-8

import os
import os.path as osp


def get_directory(ctx, filename, dirs):
    res = ctx.find_file(filename, dirs)
    return res[: -len(filename) - 1]


def check_include(ctx, use_name, folder, include_names, paths):
    # Check if required
    mandatory = False
    for req in ctx.get_env()["requires"]:
        if req == use_name:
            mandatory = True
            break

    # Generate include paths
    include_paths = []
    for path in paths:
        include_paths.append(osp.join(path, "include", folder))

    ctx.start_msg("Checking for %s includes" % str(use_name))

    for include_name in include_names:
        try:
            ctx.get_env()["INCLUDES_" + use_name] = [
                get_directory(ctx, include_name, include_paths)
            ]
            ctx.end_msg(
                "%s include found in %s"
                % (use_name, ctx.get_env()["INCLUDES_" + use_name])
            )
            break
        except:
            if mandatory:
                ctx.fatal("%s includes not found" % str(use_name))
            ctx.end_msg("%s includes not found" % str(use_name), "YELLOW")


def check_lib(ctx, use_name, lib_names, paths):
    # Check if required
    mandatory = False
    for req in ctx.get_env()["requires"]:
        if req == use_name:
            mandatory = True
            break

    # OSX/Mac uses .dylib and GNU/Linux .so
    suffix = "dylib" if ctx.env["DEST_OS"] == "darwin" else "so"

    # Generate lib paths
    lib_paths = []
    for path in paths:
        lib_paths.append(osp.join(path, "lib"))
        lib_paths.append(osp.join(path, "lib/x86_64-linux-gnu"))
        lib_paths.append(osp.join(path, "lib/intel64"))

    ctx.start_msg("Checking for %s lib" % str(use_name))

    for lib_name in lib_names:
        try:
            ctx.get_env()["LIBPATH_" + use_name] = [
                get_directory(ctx, "lib" + lib_name + "." + suffix, lib_paths)
            ]
            ctx.get_env()["LIB_" + use_name] = ctx.get_env()["LIB_" + use_name] + [
                lib_name
            ]
            ctx.end_msg(
                "%s lib found in %s" % (use_name, ctx.get_env()[
                                        "LIBPATH_" + use_name])
            )
        except:
            if mandatory:
                ctx.fatal("%s lib not found" % str(use_name))
            ctx.end_msg("%s libs not found" % str(use_name), "YELLOW")
