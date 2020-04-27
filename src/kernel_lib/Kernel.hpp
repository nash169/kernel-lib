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
#include "kernel_lib/utils/FileManager.hpp"
#include "kernel_lib/utils/Timer.hpp"

/* Tools */
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {
    /* Generic arameters */
    struct ParamsDefaults {
        struct kernel : public defaults::kernel {
        };

        struct kernel_exp : public defaults::kernel_exp {
        };

        struct kernel_cos : public defaults::kernel_cos {
        };

        struct kernel_poly : public defaults::kernel_poly {
        };

        struct kernel_expvel_dir : public defaults::kernel_expvel_dir {
            /* data */
        };

        struct expansion : public defaults::expansion {
        };
    };

    /** Exp SPHERICAL kernel
     * @default parameters defintion for kernel and expansion
     */

    using SqExp = kernels::Exp<ParamsDefaults>; // typedef kernel::Rbf<ParamsRbfSpherical> RbfSpherical;
    using SumSqExp = utils::Expansion<ParamsDefaults, SqExp>; // template <typename Params> using SumRbf = Expansion<Params, RbfSpherical>;

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

    /** Exp DIAGONAL kernel
     * @default parameters defintion for kernel and expansion
     */

    /* Diagonal covariance parameters */
    struct ParamsExpDiagonal2 {
        struct kernel : public defaults::kernel {
            PARAM_SCALAR(double, sigma_n, 1.0);
            PARAM_SCALAR(double, sigma_f, 1.0);
        };
        struct kernel_exp : public defaults::kernel_exp {
            PARAM_SCALAR(Covariance, type, CovarianceType::DIAGONAL);
            PARAM_SCALAR(bool, inverse, false);
            PARAM_VECTOR(double, sigma, 1.0, 5.0);
        };
        struct expansion : public defaults::expansion {
            PARAM_VECTOR(double, weight, 1);
        };
    };

    /* Diagonal covariance kernel */
    using ExpDiagonal2 = kernels::Exp<ParamsExpDiagonal2>;

    /* Diagonal covariance kernel expansion */
    using SumExpDiagonal2 = utils::Expansion<ParamsExpDiagonal2, ExpDiagonal2>;

    /** Exp FULL kernel
     * @default parameters defintion for kernel and expansion
     */

    /* Full covariance parameters */
    struct ParamsExpFull2 {
        struct kernel : public defaults::kernel {
            PARAM_SCALAR(double, sigma_n, 1.0);
            PARAM_SCALAR(double, sigma_f, 1.0);
        };
        struct kernel_exp : public defaults::kernel_exp {
            PARAM_SCALAR(Covariance, type, CovarianceType::FULL);
            PARAM_SCALAR(bool, inverse, false);
            PARAM_VECTOR(double, sigma, 14.5, -10.5, -10.5, 14.5); // 14.5, -10.5, -10.5, 14.5 -- 0.145, 0.105, 0.105, 0.145
        };
        struct expansion : public defaults::expansion {
            PARAM_VECTOR(double, weight, 1);
        };
    };

    /* Full covariance kernel */
    using ExpFull2 = kernels::Exp<ParamsExpFull2>;

    /* Full covariance kernel expansion */
    using SumExpFull2 = utils::Expansion<ParamsExpFull2, ExpFull2>;

} // namespace kernel_lib

#endif // KERNELLIB_KERNEL_HPP