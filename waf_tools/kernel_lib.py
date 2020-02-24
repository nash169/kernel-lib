#! /usr/bin/env python

# check if a lib exists for both osx (darwin) and GNU/linux
def check_lib(self, name, path):
    if self.env["DEST_OS"] == "darwin":
        libname = name + ".dylib"
    else:
        libname = name + ".so"
    res = self.find_file(libname, path)
    lib = res[: -len(libname) - 1]
    return lib
