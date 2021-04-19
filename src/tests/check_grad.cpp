#include <iostream>

#include <kernel_lib/Kernel.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, std::log(1.5));
        PARAM_SCALAR(double, sn, std::log(2.0));
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, std::log(0.7));
    };
};

int main(int argc, char const* argv[])
{
    std::cout << "Test" << std::endl;

    using Kernel_t = kernels::SquaredExp<Params>;
    Kernel_t k;

    size_t dim = 2, num_test = 10;

    Eigen::VectorXd val(num_test);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_test, dim),
                    y = Eigen::MatrixXd::Random(1, dim),
                    p = Eigen::MatrixXd::Random(num_test, dim) * 0.1;

    val = k(x + p, y).reshaped(num_test, 1) - k(x, y).reshaped(num_test, 1) - k.gradient(x, y).cwiseProduct(p).rowwise().sum();

    std::cout << val << std::endl;

    return 0;
}
