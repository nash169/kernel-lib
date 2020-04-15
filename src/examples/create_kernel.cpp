#include <iostream>
#include <kernel_lib/Kernel.hpp>

using namespace kernel_lib;

int main(int argc, char const* argv[])
{
    double box[] = {0, 100, 0, 100};
    size_t resolution = 100, num_samples = resolution * resolution, dim = sizeof(box) / sizeof(*box) / 2;

    Eigen::MatrixXd X(resolution, resolution), Y(resolution, resolution), x_test(num_samples, dim), x_train(3, dim);

    X = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1);
    Y = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution);

    x_test.col(0) = X.reshaped();
    x_test.col(1) = Y.reshaped();

    x_train << 25, 50,
        50, 50,
        75, 50;

    // Create covariance matrix
    Eigen::VectorXd direction(2), standard_dev(2);
    direction << 1, 1;
    standard_dev << 2, 5;
    Eigen::MatrixXd C = tools::createCovariance(direction, standard_dev);

    SumRbfSpherical psi1;
    SumRbfDiagonal2 psi2;
    SumRbfFull2 psi3;

    // Evaluate expansion for plotting
    tools::FileManager io_manager("rsc/eval_data.csv");
    io_manager.write("X", X, "Y", Y, "F", (psi1(x_train.row(0), x_test) + psi2(x_train.row(1), x_test) + psi3(x_train.row(2), x_test)).reshaped(resolution, resolution));

    return 0;
}
