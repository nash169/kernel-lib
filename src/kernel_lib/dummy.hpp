#ifndef KERNEL_LIB_DUMMY_HPP
#define KERNEL_LIB_DUMMY_HPP

#include "kernel_lib/abstract_kernel.hpp"

namespace kernel_lib {
    class Dummy {
    public:
        Dummy();

        ~Dummy();

    protected:
        double sigma_;
    };

} // namespace kernel_lib

#endif // KERNEL_LIB_DUMMY_HPP