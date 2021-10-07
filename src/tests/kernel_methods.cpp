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

    Eigen::VectorXd a(2), b(2);
    a << 0.097540404999410, 0.957506835434298;
    b << 0.970592781760616, 0.141886338627215;

    // Params
    Eigen::VectorXd params(3);
    params << std::log(1.5), std::log(2.0), std::log(0.7);

    // Eigen::VectorXd params(6);
    // params << std::log(1.5), std::log(2.0), 1.3, 0.3, 0.3, 1.1;

    std::cout << "KERNEL CREATION AND PARAMS SIZE" << std::endl;
    using Kernel_t = kernels::SquaredExp<Params>; // kernels::SquaredExpFull<Params>;
    Kernel_t k;
    std::cout << k.sizeParams() << std::endl;

    std::cout << "DEFAULT PARAMS" << std::endl;
    std::cout << k.params().transpose() << std::endl;

    std::cout << "SET PARAMS" << std::endl;
    std::cout << params.transpose() << std::endl;
    k.setParams(params);
    std::cout << k.params().transpose() << std::endl;

    std::cout << "KERNEL SINGLE POINTS xy, Xy, XX" << std::endl;
    std::cout << k(a, b) << " - " << k(x.row(0), b) << " - " << k(x.row(0), x.row(0)) << std::endl;

    std::cout << "KERNEL GRAM XY" << std::endl;
    std::cout << k.gram(x, y) << std::endl;

    std::cout << "KERNEL GRAM XX" << std::endl;
    std::cout << k.gram(x, x) << std::endl;

    std::cout << "GRADIENT SINGLE POINTS xy, Xy, XX" << std::endl;
    std::cout << k.grad(a, b).transpose() << " - " << k.grad(x.row(0), b).transpose() << " - " << k.grad(x.row(0), x.row(0)).transpose() << std::endl;

    std::cout << "GRADIENT XY" << std::endl;
    std::cout << k.gramGrad(x, y) << std::endl;

    std::cout << "GRADIENT XX" << std::endl;
    std::cout << k.gramGrad(x, x) << std::endl;

    std::cout << "HESSIAN SINGLE POINTS xy, Xy, XX" << std::endl;
    std::cout << k.hess(a, b) << std::endl;
    std::cout << " - " << std::endl;
    std::cout << k.hess(x.row(0), b) << std::endl;
    std::cout << " - " << std::endl;
    std::cout << k.hess(x.row(0), x.row(0)) << std::endl;

    std::cout << "HESSIAN XY" << std::endl;
    std::cout << k.gramHess(x, y) << std::endl;

    // std::cout << "HESSIAN XX" << std::endl;
    // std::cout << k.gramHess(x, x) << std::endl;

    std::cout << "GRADIENT PARAMS SINGLE POINTS xy, Xy, XX" << std::endl;
    std::cout << k.gradParams(a, b).transpose() << " - " << k.gradParams(x.row(0), b).transpose() << " - " << k.gradParams(x.row(0), x.row(0)).transpose() << std::endl;

    std::cout << "GRADIENT PARAMS XY" << std::endl;
    std::cout << k.gramGradParams(x, y) << std::endl;

    std::cout << "GRADIENT PARAMS XX" << std::endl;
    std::cout << k.gramGradParams(x, x) << std::endl;

    return 0;
}