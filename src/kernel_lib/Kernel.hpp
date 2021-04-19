#ifndef KERNELLIB_KERNEL_HPP
#define KERNELLIB_KERNEL_HPP

/* Kernels */
#include "kernel_lib/kernels/MaternFiveTwo.hpp"
#include "kernel_lib/kernels/MaternThreeTwo.hpp"
#include "kernel_lib/kernels/SquaredExp.hpp"
#include "kernel_lib/kernels/SquaredExpArd.hpp"
#include "kernel_lib/kernels/SquaredExpFad.hpp"
#include "kernel_lib/kernels/SquaredExpFull.hpp"

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
        struct exp_sq_ard : public defaults::exp_sq_ard {
        };
        struct exp_sq_full : public defaults::exp_sq_full {
        };
        struct exp_sq_fad : public defaults::exp_sq_fad {
        };
        struct matern_three_two : public defaults::matern_three_two {
        };
        struct matern_five_two : public defaults::matern_five_two {
        };
        struct expansion : public defaults::expansion {
        };
    };

    /** Squared Exponential Spherical Kernel and Expansion
     * @default parameters defintion for kernel and expansion
     */
    using SqExp = kernels::SquaredExp<ParamsDefaults>; // typedef kernel::Rbf<ParamsRbfSpherical> RbfSpherical;
    using SumSqExp = utils::Expansion<ParamsDefaults, SqExp>; // template <typename Params> using SumRbf = Expansion<Params, RbfSpherical>;

    /** Squared Exponential Automatic Relevance Determination Kernel and Expansion
     * @default parameters defintion for kernel and expansion
     */
    using SqExpArd = kernels::SquaredExpArd<ParamsDefaults>;
    using SumSqExpArd = utils::Expansion<ParamsDefaults, SqExpArd>;

    /** Squared Exponential Full Covariance Kernel and Expansion
     * @default parameters defintion for kernel and expansion
     */
    using SqExpFull = kernels::SquaredExpFull<ParamsDefaults>;
    using SumSqExpFull = utils::Expansion<ParamsDefaults, SqExpFull>;

    /** Squared Exponential Factor Analysis Distance Kernel and Expansion
     * @default parameters defintion for kernel and expansion
     */
    using SqExpFad = kernels::SquaredExpFad<ParamsDefaults>;
    using SumSqExpFad = utils::Expansion<ParamsDefaults, SqExpFad>;

    /** Matern Three Halves Kernel and Expansion
     * @default parameters defintion for kernel and expansion
     */
    using Matern3 = kernels::MaternThreeTwo<ParamsDefaults>;
    using SumMatern3 = utils::Expansion<ParamsDefaults, Matern3>;

    /** Matern Five Halves Kernel and Expansion
     * @default parameters defintion for kernel and expansion
     */
    using Matern5 = kernels::MaternFiveTwo<ParamsDefaults>;
    using SumMatern5 = utils::Expansion<ParamsDefaults, Matern5>;
} // namespace kernel_lib

#endif // KERNELLIB_KERNEL_HPP