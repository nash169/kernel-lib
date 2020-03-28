#include <iostream>
#include <kernel_lib/rbf.hpp>
#include <kernel_lib/tools/FileManager.hpp>

using namespace kernel_lib;

using namespace Corrade::Utility::Directory;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sigma_n, 1.0);
    };
    struct kernel_rbf : public defaults::kernel_rbf {
    };
};

int main(int argc, char const* argv[])
{
    double box[] = {0, 100, 0, 100};
    size_t resolution = 100, num_samples = resolution * resolution, dim = sizeof(box) / sizeof(*box) / 2;

    Eigen::MatrixXd X(resolution, resolution), Y(resolution, resolution), x_test(num_samples, dim), x_train(1, dim);

    X = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1);
    Y = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution);

    x_test.col(0) = X.reshaped();
    x_test.col(1) = Y.reshaped();

    x_train << 50, 50;

    Rbf<Params> k;
    {
        tools::Timer timer;
        k(x_test, x_test);
        // tools::FileManager file_to_write(join(current(), "rsc/eval_data.csv"), "w");
        // file_to_write.write("X", X, "Y", Y, "F", k(x_train, x_test).reshaped(resolution, resolution));
    }

    return 0;
}
