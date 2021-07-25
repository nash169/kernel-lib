#include <iostream>

#include <kernel_lib/Kernel.hpp>

#include <utils_cpp/UtilsCpp.hpp>

using namespace kernel_lib;

int main(int argc, char const* argv[])
{
    double box[] = {0, 100, 0, 100};
    size_t resolution = 100, num_samples = resolution * resolution, dim = sizeof(box) / sizeof(*box) / 2;

    // Data
    Eigen::MatrixXd X = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    Y = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    x_test(num_samples, dim), x_train(1, dim);
    x_test << Eigen::Map<Eigen::VectorXd>(X.data(), X.size()), Eigen::Map<Eigen::VectorXd>(Y.data(), Y.size());
    x_train << 50, 50;

    // Reference in the shape of the vector does not work for the moment
    // Eigen::Vector2d x_train = (Eigen::Vector2d() << 50, 50).finished();

    // Kernel
    using Kernel_t = kernels::SquaredExp<ParamsDefaults>;
    Kernel_t k;

    // Expansion
    using Expansion_t = utils::Expansion<ParamsDefaults, Kernel_t>;
    Expansion_t psi;
    psi.setReference(x_train);
    psi.setParams(tools::makeVector(1));

    // Solution
    utils_cpp::FileManager io_manager;
    io_manager.setFile("rsc/kernel_eval.csv");
    io_manager.write("X", X, "Y", Y, "F", psi.multiEval(x_test));

    return 0;
}