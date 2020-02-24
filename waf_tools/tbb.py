#! /usr/bin/env python
# encoding: utf-8

"""
Quick n dirty tbb detection
"""

from waflib.Configure import conf
import kernel_lib


def options(opt):
    opt.add_option("--tbb", type="string", help="path to Intel TBB", dest="tbb")


@conf
def check_tbb(self, *k, **kw):
    def get_directory(filename, dirs):
        res = self.find_file(filename, dirs)
        return res[: -len(filename) - 1]

    required = kw.get("required", False)

    if self.options.tbb:
        includes_tbb = [self.options.tbb + "/include"]
        libpath_tbb = [self.options.tbb + "/lib"]
    else:
        # If Parallel Studio XE is active use that one
        includes_tbb = ["/opt/intel/tbb/include", "/usr/local/include", "/usr/include"]
        libpath_tbb = [
            "/opt/intel/tbb/lib",
            "/usr/local/lib/",
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu/",
        ]

    self.start_msg("Checking Intel TBB includes (optional)")
    incl = ""
    lib = ""
    try:
        incl = get_directory("tbb/parallel_for.h", includes_tbb)
        self.end_msg(incl)
    except:
        if required:
            self.fatal("Not found in %s" % str(includes_tbb))
        self.end_msg("Not found in %s" % str(includes_tbb), "YELLOW")
        return

    self.start_msg("Checking Intel TBB libs (optional)")
    try:
        lib = kernel_lib.check_lib(self, "libtbb", libpath_tbb)
        self.end_msg(lib)
    except:
        if required:
            self.fatal("Not found in %s" % str(libpath_tbb))
        self.end_msg("Not found in %s" % str(libpath_tbb), "YELLOW")
        return

    self.env.LIBPATH_TBB = [lib]
    self.env.LIB_TBB = ["tbb"]
    self.env.INCLUDES_TBB = [incl]
    self.env.DEFINES_TBB = ["USE_TBB"]

