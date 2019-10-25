#ifndef KERNEL_LIB_RBF_HPP
#define KERNEL_LIB_RBF_HPP

#include "kernel_lib/kernel.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel {
        }
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class Rbf : public Kernel<Params, Rbf<Params>> {

        public:
            Rbf(/* args */) {}
            ~Rbf() {}

        private:
            double length_;
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNEL_LIB_RBF_HPP