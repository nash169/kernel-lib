#include <iostream>
#include <kernel_lib/Kernel.hpp>
#include <utils_cpp/UtilsCpp.hpp>

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

template <int size>
struct Function : kernels::SquaredExp<Params> {
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Zero(size);

    double operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return kernels::SquaredExp<Params>()(x, y);
    }
};

template <int size>
struct Gradient : kernels::SquaredExp<Params> {
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Zero(size);

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return kernels::SquaredExp<Params>::grad(x, y);
    }
};

int main(int argc, char const* argv[])
{
    using Kernel_t = kernels::SquaredExp<Params>;
    Kernel_t k;

    constexpr int dim = 2;

    Eigen::VectorXd x = Eigen::VectorXd::Random(dim);

    std::cout << "Function test" << std::endl;
    std::cout << Function<dim>()(x) << std::endl;

    std::cout << "Gradient test" << std::endl;
    std::cout << Gradient<dim>()(x).transpose() << std::endl;

    utils_cpp::DerivativeChecker checker(dim);

    if (checker.checkGradient(Function<dim>(), Gradient<dim>()))
        std::cout << "The gradient is correct!" << std::endl;

    return 0;
}
