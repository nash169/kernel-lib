#include <iostream>

#include <kernel_lib/kernels/SquaredExp.hpp>

#include <utils_cpp/UtilsCpp.hpp>

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
    double box[] = {0, 100, 0, 100};
    size_t resolution = 100, num_samples = resolution * resolution, dim = sizeof(box) / sizeof(*box) / 2;

    Eigen::MatrixXd X(resolution, resolution), Y(resolution, resolution), x_test(num_samples, dim), x_train(1, dim);

    // Test set
    X = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1);
    Y = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution);
    x_test << Eigen::Map<Eigen::VectorXd>(X.data(), X.size()), Eigen::Map<Eigen::VectorXd>(Y.data(), Y.size());

    return 0;
}