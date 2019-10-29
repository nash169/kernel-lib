#ifndef KERNEL_LIB_KERNEL_HPP
#define KERNEL_LIB_KERNEL_HPP

#include <Eigen/Core>

namespace kernel_lib {
    namespace defaults {
        struct kernel {
            BO_PARAM(double, sigma_f, 0.01);
            BO_PARAM(double, sigma_n, 1e-8);
            BO_PARAM(bool, var, true)
        };
    } // namespace defaults

    template <typename Params, typename Kernel>
    class AbstractKernel {
    public:
        AbstractKernel(const double sigma_f = Params::kernel::sigma_f(), const double sigma_n = Params::kernel::sigma_n()) : sigma_f_(sigma_f), sigma_n_(sigma_n), var(Params::kernel::var()) {}

        virtual ~AbstractKernel() {}

    private:
        double sigma_f_, sigma_n_;
        bool var;
    };

} // namespace kernel_lib

#endif // KERNEL_LIB_KERNEL_HPP