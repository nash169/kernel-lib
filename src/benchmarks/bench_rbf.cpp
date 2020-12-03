#include <iostream>

#include <kernel_lib/Kernel.hpp>

#include <utils_cpp/UtilsCpp.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        /* data */
    };

    struct rbf : public defaults::rbf {
        // Spherical
        PARAM_SCALAR(Covariance, type, CovarianceType::SPHERICAL);
        PARAM_SCALAR(bool, inverse, false);
        PARAM_VECTOR(double, sigma, 1);

        // Diagonal
        // PARAM_SCALAR(Covariance, type, CovarianceType::DIAGONAL);
        // PARAM_SCALAR(bool, inverse, false);
        // PARAM_VECTOR(double, sigma, 3, 2);

        // Full
        // PARAM_SCALAR(Covariance, type, CovarianceType::FULL);
        // PARAM_SCALAR(bool, inverse, false);
        // PARAM_VECTOR(double, sigma, 14.5, -10.5, -10.5, 14.5);

        // Full inverse
        // PARAM_SCALAR(Covariance, type, CovarianceType::FULL);
        // PARAM_SCALAR(bool, inverse, true);
        // PARAM_VECTOR(double, sigma, 0.145, 0.105, 0.105, 0.145);
    };
};

int main(int argc, char const* argv[])
{
    size_t dim = 2, num_samples = 20000;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, dim);

    using Kernel_t = kernels::Rbf<Params>;

    Kernel_t k;
    {
        utils_cpp::Timer timer;
        k(X, X);
    }

    return 0;
}