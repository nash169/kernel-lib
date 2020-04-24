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
    /** Define elements based on spherical covariance RBF */
    struct ParamsExpSpherical {
        struct kernel : public defaults::kernel {
            PARAM_SCALAR(double, sigma_n, 1.0);
            PARAM_SCALAR(double, sigma_f, 1.0);
        };
        struct kernel_exp : public defaults::kernel_exp {
            PARAM_SCALAR(Covariance, type, CovarianceType::SPHERICAL);
            PARAM_SCALAR(bool, inverse, false);
            PARAM_VECTOR(double, sigma, 1.0);
        };
        struct expansion : public defaults::expansion {
            PARAM_VECTOR(double, weight, 1);
        };
    };

    // Spherical RBF
    using ExpSpherical = kernels::Exp<ParamsExpSpherical>;
    // typedef kernel::Rbf<ParamsRbfSpherical> RbfSpherical;

    // Kernel expansion based on spherical RBF
    using SumExpSpherical = utils::Expansion<ParamsExpSpherical, ExpSpherical>;
    // template <typename Params>
    // using SumRbf = Expansion<Params, RbfSpherical>;

    /** Define elements based on diagonal covariance RBF */
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

    // Diagonal RBF
    using ExpDiagonal2 = kernels::Exp<ParamsExpDiagonal2>;
    // Kernel expansion based on spherical RBF
    using SumExpDiagonal2 = utils::Expansion<ParamsExpDiagonal2, ExpDiagonal2>;

    /** Define elements based on full covariance RBF */
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

    // Diagonal RBF
    using ExpFull2 = kernels::Exp<ParamsExpFull2>;
    // Kernel expansion based on spherical RBF
    using SumExpFull2 = utils::Expansion<ParamsExpFull2, ExpFull2>;

} // namespace kernel_lib

#endif // KERNELLIB_KERNEL_HPP