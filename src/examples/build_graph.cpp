#include <iostream>

#include <kernel_lib/Kernel.hpp>

using namespace kernel_lib;

// Kernel parameters
struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.405465);
        PARAM_SCALAR(double, sn, 0.693147);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, -0.356675);
    };
};

int main(int argc, char const* argv[])
{
    constexpr int dim = 2, num_samples = 10;

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, dim);

    utils::Graph graph;

    auto mat = graph.kNearest(X, 3, 1);

    std::cout << mat.toDense() << std::endl;

    auto matW = graph.kNearestWeighted(X, kernels::SquaredExp<Params>(), 3, 1);

    std::cout << matW.toDense() << std::endl;

    return 0;
}