#include <iostream>

#include <kernel_lib/Kernel.hpp>
#include <utils_cpp/UtilsCpp.hpp>

using namespace kernel_lib;

int main(int argc, char const* argv[])
{
    double box[] = {0, 100, 0, 100};
    size_t resolution = 100, num_samples = resolution * resolution, dim = sizeof(box) / sizeof(*box) / 2;

    Eigen::MatrixXd X(resolution, resolution), Y(resolution, resolution), x_test(num_samples, dim), x_train(1, dim);

    // Test set
    X = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1);
    Y = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution);
    x_test.col(0) = X.reshaped();
    x_test.col(1) = Y.reshaped();

    // Reference set
    x_train << 50, 50;

    // File manager
    utils_cpp::FileManager io_manager;

    // Squared Exponential kernel (Spherical covariance)
    std::cout << "Square Exponential kernel (Spherical)" << std::endl;
    SumSqExp rbf_spherical;

    io_manager.setFile("rsc/rbf_spherical.csv");
    io_manager.write("X", X, "Y", Y, "F", rbf_spherical(x_train, x_test));

    // Squared Exponential kernel (Diagonal covariance)
    std::cout << "Square Exponential kernel (Diagonal)" << std::endl;
    SumSqExpArd rbf_diagonal;

    io_manager.setFile("rsc/rbf_diagonal.csv");
    io_manager.write("X", X, "Y", Y, "F", rbf_diagonal(x_train, x_test));

    // Squared Exponential kernel (Full covariance)
    std::cout << "Square Exponential kernel (Full)" << std::endl;
    SumSqExpFull rbf_full;

    io_manager.setFile("rsc/rbf_full.csv");
    io_manager.write("X", X, "Y", Y, "F", rbf_full(x_train, x_test));

    return 0;
}
