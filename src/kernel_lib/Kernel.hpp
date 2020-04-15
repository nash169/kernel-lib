#ifndef KERNELLIB_KERNEL_HPP
#define KERNELLIB_KERNEL_HPP

#include "kernel_lib/Expansion.hpp"
#include "kernel_lib/kernel/Rbf.hpp"
#include "kernel_lib/tools/FileManager.hpp"
#include "kernel_lib/tools/Timer.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {
    /** Define elements based on spherical covariance RBF */
    struct ParamsRbfSpherical {
        struct kernel : public defaults::kernel {
            PARAM_SCALAR(double, sigma_n, 1.0);
            PARAM_SCALAR(double, sigma_f, 1.0);
        };
        struct kernel_rbf : public defaults::kernel_rbf {
            PARAM_SCALAR(Covariance, type, CovarianceType::SPHERICAL);
            PARAM_SCALAR(bool, inverse, false);
            PARAM_VECTOR(double, sigma, 1.0);
        };
        struct expansion : public defaults::expansion {
            PARAM_VECTOR(double, weight, 1);
        };
    };

    // Spherical RBF
    using RbfSpherical = kernel::Rbf<ParamsRbfSpherical>;
    // typedef kernel::Rbf<ParamsRbfSpherical> RbfSpherical;

    // Kernel expansion based on spherical RBF
    using SumRbfSpherical = Expansion<ParamsRbfSpherical, RbfSpherical>;
    // template <typename Params>
    // using SumRbf = Expansion<Params, RbfSpherical>;

    /** Define elements based on diagonal covariance RBF */
    struct ParamsRbfDiagonal2 {
        struct kernel : public defaults::kernel {
            PARAM_SCALAR(double, sigma_n, 1.0);
            PARAM_SCALAR(double, sigma_f, 1.0);
        };
        struct kernel_rbf : public defaults::kernel_rbf {
            PARAM_SCALAR(Covariance, type, CovarianceType::DIAGONAL);
            PARAM_SCALAR(bool, inverse, false);
            PARAM_VECTOR(double, sigma, 1.0, 5.0);
        };
        struct expansion : public defaults::expansion {
            PARAM_VECTOR(double, weight, 1);
        };
    };

    // Diagonal RBF
    using RbfDiagonal2 = kernel::Rbf<ParamsRbfDiagonal2>;
    // Kernel expansion based on spherical RBF
    using SumRbfDiagonal2 = Expansion<ParamsRbfDiagonal2, RbfDiagonal2>;

    /** Define elements based on full covariance RBF */
    struct ParamsRbfFull2 {
        struct kernel : public defaults::kernel {
            PARAM_SCALAR(double, sigma_n, 1.0);
            PARAM_SCALAR(double, sigma_f, 1.0);
        };
        struct kernel_rbf : public defaults::kernel_rbf {
            PARAM_SCALAR(Covariance, type, CovarianceType::FULL);
            PARAM_SCALAR(bool, inverse, false);
            PARAM_VECTOR(double, sigma, 14.5, -10.5, -10.5, 14.5); // 14.5, -10.5, -10.5, 14.5 -- 0.145, 0.105, 0.105, 0.145
        };
        struct expansion : public defaults::expansion {
            PARAM_VECTOR(double, weight, 1);
        };
    };

    // Diagonal RBF
    using RbfFull2 = kernel::Rbf<ParamsRbfFull2>;
    // Kernel expansion based on spherical RBF
    using SumRbfFull2 = Expansion<ParamsRbfFull2, RbfFull2>;

} // namespace kernel_lib

#endif // KERNELLIB_KERNEL_HPP