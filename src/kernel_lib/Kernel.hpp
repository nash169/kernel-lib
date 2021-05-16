#ifndef KERNELLIB_KERNEL_HPP
#define KERNELLIB_KERNEL_HPP

/* Kernels */
#include "kernel_lib/kernels/SquaredExp.hpp"

/* Utils */
#include "kernel_lib/utils/Expansion.hpp"

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
        struct expansion : public defaults::expansion {
        };
    };

    /** Squared Exponential Spherical Kernel and Expansion
     * @default parameters defintion for kernel and expansion
     */
    using SqExp = kernels::SquaredExp<ParamsDefaults>; // typedef kernel::Rbf<ParamsRbfSpherical> RbfSpherical;
    using SumSqExp = utils::Expansion<ParamsDefaults, SqExp>; // template <typename Params> using SumRbf = Expansion<Params, RbfSpherical>;
} // namespace kernel_lib

#endif // KERNELLIB_KERNEL_HPP