#include <iostream>

#include <kernel_lib/Kernel.hpp>

#include <utils_cpp/UtilsCpp.hpp>

using namespace kernel_lib;

int main(int argc, char const* argv[])
{
    double box[] = {0, 10, 0, 10};
    size_t resolution = 100, num_samples = resolution * resolution, dim = sizeof(box) / sizeof(*box) / 2;

    // Data
    Eigen::MatrixXd X = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    Y = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    x_test(num_samples, dim), mu(1, dim);
    x_test << Eigen::Map<Eigen::VectorXd>(X.data(), X.size()), Eigen::Map<Eigen::VectorXd>(Y.data(), Y.size());
    mu << 5, 5;
    Eigen::Vector4d cov(1, 0.5, 0.5, 1);

    // Kernel
    // using Kernel_t = kernels::SquaredExp<ParamsDefaults>;
    using Kernel_t = kernels::SquaredExpFull<ParamsDefaults>;
    Kernel_t k;

    // Gaussian
    using Gaussian_t = utils::Gaussian<ParamsDefaults, Kernel_t>;
    Gaussian_t gauss;

    // First covariance matrix then mean
    // gauss.setParams((Eigen::Vector3d() << 1, mu.transpose()).finished());
    gauss.setParams((Eigen::Matrix<double, 6, 1>() << cov, mu.transpose()).finished());

    // Solution
    utils_cpp::FileManager io_manager;
    io_manager.setFile("rsc/gauss_eval.csv");
    io_manager.write("X", X, "Y", Y, "F", gauss.multiEval(x_test));

    return 0;
}