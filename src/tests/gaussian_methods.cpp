#include <iostream>
#include <kernel_lib/Kernel.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.5);
        PARAM_SCALAR(double, sn, 1.4);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, 3.1);
    };

    struct exp_sq_full {
        PARAM_VECTOR(double, S, 1, 0.5, 0.5, 1);
    };

    struct gaussian {
        PARAM_VECTOR(double, mu, 3.4, 10.1);
    };
};

int main(int argc, char const* argv[])
{
    // Data
    Eigen::Vector2d x(0.097540404999410, 0.957506835434298);
    // Eigen::Vector3d params(-0.356675, 2.0, 0.7);
    Eigen::Matrix<double, 6, 1> params(1.3, 0.3, 0.3, 1.1, 43.2, 12.6);

    // Gaussian
    using Kernel_t = kernels::SquaredExpFull<Params>;
    using Gaussian_t = utils::Gaussian<Params, Kernel_t>;
    Gaussian_t gauss;

    std::cout << "Gaussian -> params()" << std::endl;
    std::cout << gauss.params().transpose() << std::endl;

    std::cout << "Gaussian -> setParams()" << std::endl;
    gauss.setParams(params);
    std::cout << gauss.params().transpose() << std::endl;

    std::cout << "Gaussian -> sizeParams()" << std::endl;
    std::cout << gauss.sizeParams() << std::endl;

    std::cout << "Gaussian -> operator()" << std::endl;
    std::cout << gauss(x) << std::endl;

    std::cout << "Gaussian -> log()" << std::endl;
    std::cout << gauss.log(x) << std::endl;

    std::cout << "Gaussian -> grad()" << std::endl;
    std::cout << gauss.grad(x).transpose() << std::endl;

    std::cout << "Gaussian -> gradParams()" << std::endl;
    std::cout << gauss.gradParams(x).transpose() << std::endl;

    return 0;
}