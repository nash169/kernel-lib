#include <iostream>
#include <kernel_lib/Kernel.hpp>

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
    utils::FileManager io_manager;

    // Squared Exponential kernel (Spherical covariance)
    std::cout << "Square Exponential kernel (Spherical)" << std::endl;
    SumSqExp exp_spherical;

    io_manager.setFile("rsc/kernel_eval/exp_spherical.csv");
    io_manager.write("X", X, "Y", Y, "F", exp_spherical(x_train, x_test).reshaped(resolution, resolution));

    // Squared Exponential kernel (Diagonal covariance)
    std::cout << "Square Exponential kernel (Diagonal)" << std::endl;
    SumSqExp exp_diagonal;

    Eigen::VectorXd params_diag(2);
    params_diag << 1, 5;
    exp_diagonal.kernel().setCovariance(CovarianceType::DIAGONAL).setParameters(params_diag);

    io_manager.setFile("rsc/kernel_eval/exp_diagonal.csv");
    io_manager.write("X", X, "Y", Y, "F", exp_diagonal(x_train, x_test).reshaped(resolution, resolution));

    // Squared Exponential kernel (Full covariance)
    std::cout << "Square Exponential kernel (Full)" << std::endl;
    SumSqExp exp_full;

    Eigen::VectorXd direction(2), standard_dev(2);
    direction << 1, 1;
    standard_dev << 2, 5;
    Eigen::MatrixXd C = tools::createCovariance(direction, standard_dev);
    exp_full.kernel().setCovariance(CovarianceType::FULL).setParameters(C.reshaped(std::pow(dim, 2), 1));

    io_manager.setFile("rsc/kernel_eval/exp_full.csv");
    io_manager.write("X", X, "Y", Y, "F", exp_full(x_train, x_test).reshaped(resolution, resolution));

    // Cosine kernel
    std::cout << "Cosine kernel" << std::endl;
    SumCosine cosine;

    io_manager.setFile("rsc/kernel_eval/cosine.csv");
    io_manager.write("X", X, "Y", Y, "F", cosine(x_train, x_test).reshaped(resolution, resolution));

    // Polynomial kernel
    std::cout << "Polynomial kernel" << std::endl;
    SumPolynomial polynomial;

    io_manager.setFile("rsc/kernel_eval/polynomial.csv");
    io_manager.write("X", X, "Y", Y, "F", polynomial(x_train, x_test).reshaped(resolution, resolution));

    return 0;
}
