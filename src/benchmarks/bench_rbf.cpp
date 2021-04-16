#include <iostream>

#include <kernel_lib/Kernel.hpp>

#include <utils_cpp/UtilsCpp.hpp>

#include <kernel_lib/kernels/SquaredExp.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sigma_n, 2.0);
        PARAM_SCALAR(double, sigma_f, 1.5);
    };

    struct kernel2 : public defaults::kernel2 {
        PARAM_SCALAR(double, sf, std::log(0.5));
        PARAM_SCALAR(double, sn, std::log(1.4));
    };

    struct rbf : public defaults::rbf {
        // Spherical
        PARAM_SCALAR(Covariance, type, CovarianceType::SPHERICAL);
        PARAM_VECTOR(double, sigma, 0.7);

        // Diagonal
        // PARAM_SCALAR(Covariance, type, CovarianceType::DIAGONAL);
        // PARAM_VECTOR(double, sigma, 0.7, 1.3);

        // Full
        // PARAM_SCALAR(Covariance, type, CovarianceType::FULL);
        // PARAM_VECTOR(double, sigma, 14.5, -10.5, -10.5, 14.5);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, std::log(3.1));
    };
};

int main(int argc, char const* argv[])
{
    // // using Kernel_t = kernels::Rbf<Params>;
    // using Kernel_t = kernels::SquaredExp<Params>;
    // Kernel_t k;

    // size_t dim = 2, num_samples = 10000;
    // Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, dim), Y = Eigen::MatrixXd::Random(num_samples, dim);

    // {
    //     utils_cpp::Timer timer;
    //     Eigen::MatrixXd test = k.gradient(X, X, 1);
    // }

    // using Kernel_t = kernels::Rbf<Params>;

    // Data
    Eigen::MatrixXd x(3, 2), y(4, 2);

    x << 0.097540404999410, 0.957506835434298,
        0.278498218867048, 0.964888535199277,
        0.546881519204984, 0.157613081677548;

    y << 0.970592781760616, 0.141886338627215,
        0.957166948242946, 0.421761282626275,
        0.485375648722841, 0.915735525189067,
        0.800280468888800, 0.792207329559554;

    // Params
    Eigen::VectorXd params(3);
    params << std::log(1.5), std::log(2.0), std::log(0.7);

    std::cout << "KERNEL CREATION AND PARAMS SIZE" << std::endl;
    using Kernel_t = kernels::SquaredExp<Params>;
    Kernel_t k;
    std::cout << k.sizeParams() << std::endl;

    std::cout << "DEFAULT PARAMS" << std::endl;
    std::cout << k.params().transpose() << std::endl;

    std::cout << "SET PARAMS" << std::endl;
    std::cout << params.transpose() << std::endl;
    k.setParams(params);
    std::cout << k.params().transpose() << std::endl;

    std::cout << "KERNEL" << std::endl;
    std::cout << k(x, y) << std::endl;

    std::cout << "GRADIENT" << std::endl;
    std::cout << k.gradient(x, y) << std::endl;

    std::cout << "GRADIENT PARAMS" << std::endl;
    std::cout << k.gradientParams(x, y) << std::endl;

    return 0;
}