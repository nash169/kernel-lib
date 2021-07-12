#include <iostream>
#include <kernel_lib/Kernel.hpp>

#define KERNEL kernels::SquaredExp<Params>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.5);
        PARAM_SCALAR(double, sn, 1.4);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, 3.1);
    };
};

int main(int argc, char const* argv[])
{
    // Data
    Eigen::MatrixXd x(3, 2), x_i(5, 2);
    x << 0.097540404999410, 0.957506835434298,
        0.278498218867048, 0.964888535199277,
        0.546881519204984, 0.157613081677548;

    x_i << 0.970592781760616, 0.141886338627215,
        0.957166948242946, 0.421761282626275,
        0.485375648722841, 0.915735525189067,
        0.800280468888800, 0.792207329559554,
        0.097540404999410, 0.278498218867048;

    Eigen::VectorXd a(2);
    a << 0.097540404999410, 0.957506835434298;

    // Params
    Eigen::VectorXd params(3);
    params << std::log(1.5), std::log(2.0), std::log(0.7);

    // Weights
    Eigen::VectorXd weights(x_i.rows());
    weights << 0.823457828327293, 0.694828622975817, 0.317099480060861, 0.950222048838355, 0.034446080502909;

    std::cout << "EXPANSION CREATION";
    using Kernel_t = KERNEL;
    using Expansion_t = utils::Expansion<Params, Kernel_t>;
    Expansion_t f;
    f.kernel().setParams(params);
    std::cout << "DONE";

    std::cout << "EXPANSION PARAMS" << std::endl;
    f.setParams(weights);
    std::cout << f.params().transpose() << std::endl;

    std::cout << "EXPANSION REFERENCE" << std::endl;
    f.setReference(x_i);
    std::cout << f.reference().transpose() << std::endl;

    std::cout << "EXPANSION SINGLE POINT" << std::endl;
    std::cout << f(a) << std::endl;

    std::cout << "EXPANSION MULTIPLE POINT" << std::endl;
    std::cout << f.multiEval(x).transpose() << std::endl;

    return 0;
}