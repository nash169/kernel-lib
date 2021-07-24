#ifndef KERNELLIB_KERNEL_HPP
#define KERNELLIB_KERNEL_HPP

/* Squared Exponential Kernel Spherical Covariance */
#include "kernel_lib/kernels/SquaredExp.hpp"

/* Squared Exponential Kernel Full Covariance */
#include "kernel_lib/kernels/SquaredExpFull.hpp"

/* Riemannian Squared Exponential Kernel */
#include <kernel_lib/kernels/RiemannSqExp.hpp>

/* Riemannian Matern Kernel */
// #include <kernel_lib/kernels/RiemannMatern.hpp>

/* Kernel expansion */
#include "kernel_lib/utils/Expansion.hpp"

/* Build Graph */
#include "kernel_lib/utils/Graph.hpp"

/* Gaussian distribution */
#include "kernel_lib/utils/Gaussian.hpp"

/* Tools */
#include "kernel_lib/tools/helper.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {
    /* Generic arameters */
    struct ParamsDefaults {
        struct kernel : public defaults::kernel {
        };
        struct exp_sq : public defaults::exp_sq {
        };
        struct exp_sq_full : public defaults::exp_sq_full {
        };
        struct riemann_exp_sq : public defaults::riemann_exp_sq {
        };
        // struct riemann_matern : public defaults::riemann_matern {
        // };
        struct expansion : public defaults::expansion {
        };
        struct gaussian : public defaults::gaussian {
        };
    };

    /** Squared Exponential Spherical Kernel and Expansion
     * @default parameters defintion for kernel and expansion
     */
    using SqExp = kernels::SquaredExp<ParamsDefaults>; // typedef kernel::Rbf<ParamsRbfSpherical> RbfSpherical;
    using SumSqExp = utils::Expansion<ParamsDefaults, SqExp>; // template <typename Params> using SumRbf = Expansion<Params, RbfSpherical>;
} // namespace kernel_lib

#endif // KERNELLIB_KERNEL_HPP