#include <iostream>
#include <kernel_lib/Kernel.hpp>
#include <utils_cpp/UtilsCpp.hpp>

#define KERNEL kernels::SquaredExp<Params>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.405465);
        PARAM_SCALAR(double, sn, 0.693147);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, -0.356675);
    };
};

template <int size>
struct FunctionX : public KERNEL {
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Zero(size);

    double operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return KERNEL()(x, y);
    }
};

template <int size>
struct GradientX : public KERNEL {
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Zero(size);

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return KERNEL::grad(x, y);
    }
};

template <int size>
struct FunctionY : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

    double operator()(const Eigen::Matrix<double, size, 1>& y) const
    {
        return KERNEL()(x, y);
    }
};

template <int size>
struct GradientY : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& y) const
    {
        return KERNEL::grad(x, y, 0);
    }
};

int main(int argc, char const* argv[])
{
    using Kernel_t = KERNEL;
    Kernel_t k;

    constexpr int dim = 2;

    Eigen::VectorXd x = Eigen::VectorXd::Random(dim);

    std::cout << "Function in X test" << std::endl;
    std::cout << FunctionX<dim>()(x) << std::endl;

    std::cout << "Gradient in X test" << std::endl;
    std::cout << GradientX<dim>()(x).transpose() << std::endl;

    std::cout << "Function in Y test" << std::endl;
    std::cout << FunctionY<dim>()(x) << std::endl;

    std::cout << "Gradient in Y test" << std::endl;
    std::cout << GradientY<dim>()(x).transpose() << std::endl;

    utils_cpp::DerivativeChecker checker(dim);

    std::cout << "Checking the gradient with respect to X..." << std::endl;
    if (checker.checkGradient(FunctionX<dim>(), GradientX<dim>()))
        std::cout << "The gradient is CORRECT!" << std::endl;
    else
        std::cout << "The gradient is NOT correct!" << std::endl;

    std::cout << "Checking the gradient with respect to Y..." << std::endl;
    if (checker.checkGradient(FunctionY<dim>(), GradientY<dim>()))
        std::cout << "The gradient is CORRECT!" << std::endl;
    else
        std::cout << "The gradient is NOT correct!" << std::endl;

    return 0;
}
