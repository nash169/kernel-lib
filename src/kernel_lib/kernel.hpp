#ifndef KERNEL_LIB_KERNEL_HPP
#define KERNEL_LIB_KERNEL_HPP

#include "kernel_lib/tools/macros.hpp"
#include <Eigen/Core>

namespace kernel_lib {
    namespace defaults {
        struct kernel {
            /// @ingroup kernel_defaults
            BO_PARAM(double, noise, 0.01);
            BO_PARAM(bool, optimize_noise, false);
        };
    } // namespace defaults

    template <typename Params>
    class Kernel {
    public:
        Kernel(bool optim_noise = Params::kernel::optimize_noise(), double var = 0.5) : noise_(Params::kernel::noise()), var_(var)
        {
            optim_noise ? std::cout << "hello" << std::endl : std::cout << "ciao" << std::endl;
        }
        ~Kernel() {}

        double noise() { return noise_; }

        double var() { return var_; }

    private:
        double noise_,
            var_;
    };

} // namespace kernel_lib

#endif // KERNEL_LIB_KERNEL_HPP