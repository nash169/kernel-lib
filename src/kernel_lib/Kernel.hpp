#ifndef KERNELLIB_KERNEL_HPP
#define KERNELLIB_KERNEL_HPP

/* Kernels */
#include "kernel_lib/kernels/Rbf.hpp"

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

        struct rbf : public defaults::rbf {
        };

        struct expansion : public defaults::expansion {
        };
    };

    /** Exp spherical kernel
     * @default parameters defintion for kernel and expansion
     */

    using Rbf = kernels::Rbf<ParamsDefaults>; // typedef kernel::Rbf<ParamsRbfSpherical> RbfSpherical;
    using SumRbf = utils::Expansion<ParamsDefaults, Rbf>; // template <typename Params> using SumRbf = Expansion<Params, RbfSpherical>;
} // namespace kernel_lib

#endif // KERNELLIB_KERNEL_HPP