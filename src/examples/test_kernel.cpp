#include <iostream>

#include <kernel_lib/Expansion.hpp>
#include <kernel_lib/kernel/Rbf.hpp>

#include <kernel_lib/tools/FileManager.hpp>
#include <kernel_lib/tools/Timer.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sigma_n, 1.0);
    };
    struct kernel_rbf : public defaults::kernel_rbf {
        PARAM_VECTOR(double, sigma, 5);
    };
    struct expansion : public defaults::expansion {
        PARAM_VECTOR(double, weight, 5, 3);
    };
};

int main(int argc, char const* argv[])
{
    double box[] = {0, 100, 0, 100};
    size_t resolution = 100, num_samples = resolution * resolution, dim = sizeof(box) / sizeof(*box) / 2;

    Eigen::MatrixXd X(resolution, resolution), Y(resolution, resolution), x_test(num_samples, dim), x_train(2, dim);

    X = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1);
    Y = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution);

    x_test.col(0) = X.reshaped();
    x_test.col(1) = Y.reshaped();

    x_train << 25, 25,
        75, 75;

    // the type of the GP
    using Kernel_t = kernel::Rbf<Params>;
    using Expansion_t = Expansion<Params, Kernel_t>;

    // Evaluate expansion for plotting
    Expansion_t psi;
    tools::FileManager file_to_write(join(current(), "rsc/eval_data.csv"), "w");
    file_to_write.write("X", X, "Y", Y, "F", psi(x_train, x_test).reshaped(resolution, resolution));

    // Benchmark kernel
    Kernel_t k;
    {
        tools::Timer timer;
        k(x_test, x_test);
    }

    return 0;
}
