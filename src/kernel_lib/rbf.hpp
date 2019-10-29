#ifndef KERNEL_LIB_RBF_HPP
#define KERNEL_LIB_RBF_HPP

#include "kernel_lib/abstract_kernel.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel_rbf {
            BO_PARAM(double, sigma, 1);
        };
    } // namespace defaults

    template <typename Params>
    class Rbf : public AbstractKernel<Params, Rbf<Params>> {
    public:
        Rbf(const double sigma = Params::kernel_rbf::sigma()) : sigma_(sigma) {}
        ~Rbf() {}

    private:
        double sigma_;
    };

} // namespace kernel_lib

#endif // KERNEL_LIB_RBF_HPP