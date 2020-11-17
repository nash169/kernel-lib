#ifndef KERNELLIB_KERNEL_HPP
#define KERNELLIB_KERNEL_HPP

/* Kernels */
#include "kernel_lib/kernels/Cosine.hpp"
#include "kernel_lib/kernels/Exp.hpp"
#include "kernel_lib/kernels/ExpGradientDirected.hpp"
#include "kernel_lib/kernels/ExpVelocityDirected.hpp"
#include "kernel_lib/kernels/Polynomial.hpp"

/* Utils */
#include "kernel_lib/utils/Expansion.hpp"

/* Tools */
#include "kernel_lib/tools/helper.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {
    /* Generic arameters */
    struct ParamsDefaults {
        struct kernel_exp : public defaults::kernel_exp {
        };

        struct kernel_poly : public defaults::kernel_poly {
        };

        struct kernel_cos : public defaults::kernel_cos {
        };

        struct kernel_exp_vel : public defaults::kernel_exp_vel {
        };

        struct expansion : public defaults::expansion {
        };
    };

    /** Exp spherical kernel
     * @default parameters defintion for kernel and expansion
     */

    using Exp = kernels::Exp<ParamsDefaults>; // typedef kernel::Rbf<ParamsRbfSpherical> RbfSpherical;
    using SumExp = utils::Expansion<ParamsDefaults, Exp>; // template <typename Params> using SumRbf = Expansion<Params, RbfSpherical>;

    /** Cosine kernel
     * @default parameters defintion for kernel and expansion
     */

    typedef kernels::Cosine<ParamsDefaults> Cosine;
    typedef utils::Expansion<ParamsDefaults, Cosine> SumCosine;

    /** Polynomial kernel
     * @default parameters defintion for kernel and expansion
     */

    typedef kernels::Polynomial<ParamsDefaults> Polynomial;
    typedef utils::Expansion<ParamsDefaults, Polynomial> SumPolynomial;

    /** Squared Exponential Velocity Directed kernel
     * @default parameters defintion for kernel and expansion
     */

    typedef kernels::ExpVelocityDirected<ParamsDefaults> ExpVelocityDirected;
    typedef utils::Expansion<ParamsDefaults, ExpVelocityDirected> SumExpVelocityDirected;
} // namespace kernel_lib

#endif // KERNELLIB_KERNEL_HPP