#include <chrono>
#include <iostream>
#include <thread>

#include <kernel_lib/Kernel.hpp>
#include <utils_cpp/UtilsCpp.hpp>

// #include <kernel_lib/kernels/SquaredExp2.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, std::log(0.5));
        PARAM_SCALAR(double, sn, std::log(1.4));
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, std::log(3.1));
    };
};

int main(int argc, char const* argv[])
{
    // using Kernel_t = kernels::SquaredExp2<Params>;
    using Kernel_t = kernels::SquaredExp<Params>;
    Kernel_t k;

    int dim = 2, num_samples = 20000;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, dim), Y = Eigen::MatrixXd::Random(num_samples, dim);

    std::cout << "BENCHMARK: Kernel evaluation" << std::endl;
    {
        utils_cpp::Timer timer;
        k(X, X);
    }

    return 0;
}