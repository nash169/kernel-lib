#include <chrono>
#include <iostream>
#include <thread>

#include <kernel_lib/Kernel.hpp>

#include <utils_cpp/UtilsCpp.hpp>

#include <kernel_lib/kernels/SquaredExp.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel2 : public defaults::kernel2 {
        PARAM_SCALAR(double, sf, std::log(0.5));
        PARAM_SCALAR(double, sn, std::log(1.4));
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, std::log(3.1));
    };
};

int main(int argc, char const* argv[])
{
    std::chrono::seconds wait(2);

    // using Kernel_t = kernels::Rbf<Params>;
    using Kernel_t = kernels::SquaredExp<Params>;
    Kernel_t k;

    size_t dim = 2, num_samples = 10000;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, dim), Y = Eigen::MatrixXd::Random(num_samples, dim);

    std::cout << "BENCHMARK: Kernel evaluation" << std::endl;
    {
        utils_cpp::Timer timer;
        k(X, X);
    }

    std::this_thread::sleep_for(wait);

    std::cout << "BENCHMARK: Kernel gradient" << std::endl;
    {
        utils_cpp::Timer timer;
        k.gradient(X, X);
    }

    std::this_thread::sleep_for(wait);

    std::cout << "BENCHMARK: Kernel params gradient" << std::endl;
    {
        utils_cpp::Timer timer;
        k.gradientParams(X, X);
    }

    return 0;
}