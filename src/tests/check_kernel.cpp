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

    struct exp_sq_ard : public defaults::exp_sq_ard {
        PARAM_VECTOR(double, l, 1.1, 0.6);
    };

    struct exp_sq_full : public defaults::exp_sq_full {
        PARAM_VECTOR(double, l, 3.1, 4.5, 0.8, 1.2);
    };
};

int main(int argc, char const* argv[])
{
    // Data
    Eigen::MatrixXd x(3, 2), y(5, 2);

    x << 0.097540404999410, 0.957506835434298,
        0.278498218867048, 0.964888535199277,
        0.546881519204984, 0.157613081677548;

    y << 0.970592781760616, 0.141886338627215,
        0.957166948242946, 0.421761282626275,
        0.485375648722841, 0.915735525189067,
        0.800280468888800, 0.792207329559554,
        0.097540404999410, 0.278498218867048;

    // Params
    Eigen::VectorXd params(3);
    params << std::log(1.5), std::log(2.0), std::log(0.7);
    // params << std::log(1.5), std::log(2.0), std::log(0.7), std::log(1.3);
    // params << std::log(1.5), std::log(2.0), 6.5, 2.5, 2.5, 6.5;

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