#include <iostream>

#include <kernel_lib/Kernel.hpp>

#include <utils_cpp/UtilsCpp.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sigma_n, 2.0);
        PARAM_SCALAR(double, sigma_f, 1.5);
    };

    struct rbf : public defaults::rbf {
        // Spherical
        PARAM_SCALAR(Covariance, type, CovarianceType::SPHERICAL);
        PARAM_VECTOR(double, sigma, 1);

        // Diagonal
        // PARAM_SCALAR(Covariance, type, CovarianceType::DIAGONAL);
        // PARAM_VECTOR(double, sigma, 3, 2);

        // Full
        // PARAM_SCALAR(Covariance, type, CovarianceType::FULL);
        // PARAM_VECTOR(double, sigma, 14.5, -10.5, -10.5, 14.5);
    };
};

int main(int argc, char const* argv[])
{
    size_t dim = 2, num_samples = 20000;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, dim), Y = Eigen::MatrixXd::Random(num_samples, dim);

    // Eigen::MatrixXd X(4, 2);

    // X << 0.097540404999410, 0.964888535199277,
    //     0.278498218867048, 0.157613081677548,
    //     0.546881519204984, 0.970592781760616,
    //     0.957506835434298, 0.957166948242946;

    using Kernel_t = kernels::Rbf<Params>;
    Kernel_t k;

    // std::cout << k(X, X) << std::endl;

    {
        utils_cpp::Timer timer;
        k(X, X);
    }

    return 0;
}