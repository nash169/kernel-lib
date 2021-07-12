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
struct FunctionT : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    double operator()(const Eigen::VectorXd& params)
    {
        KERNEL::setParams(params);

        return KERNEL::operator()(x, y);
    }
};

template <int size>
struct GradientT : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    Eigen::VectorXd operator()(const Eigen::VectorXd& params)
    {
        KERNEL::setParams(params);

        return KERNEL::gradParams(x, y);
    }
};

template <int size>
struct FunctionS : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    double operator()(const Eigen::VectorXd& params)
    {
        KERNEL::setParameters(params);

        return KERNEL::kernel(x, y);
    }
};

template <int size>
struct GradientS : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    Eigen::VectorXd operator()(const Eigen::VectorXd& params)
    {
        KERNEL::setParameters(params);

        return tools::makeVector(KERNEL::gradientParams(x, y));
    }
};

int main(int argc, char const* argv[])
{
    using Kernel_t = KERNEL;
    Kernel_t k;

    constexpr int dim = 2;

    Eigen::Matrix<double, dim, 1> x = Eigen::VectorXd::Zero(dim);
    Eigen::Matrix<double, dim, 1> y = Eigen::VectorXd::Ones(dim);

    Eigen::VectorXd params_t = Eigen::VectorXd::Random(k.sizeParams()), params_s = Eigen::VectorXd::Random(k.sizeParams() - 2);

    std::cout << "Function TOTAL kernel test" << std::endl;
    std::cout << FunctionT<dim>()(params_t) << std::endl;
    k.setParams(params_t);
    std::cout << k(x, y) << std::endl;

    std::cout << "Gradient TOTAL kernel test" << std::endl;
    std::cout << GradientT<dim>()(params_t).transpose() << std::endl;

    std::cout << "Function SPECIFIC kernel test" << std::endl;
    std::cout << FunctionS<dim>()(params_s) << std::endl;
    std::cout << k.kernel(x, y) << std::endl;

    std::cout << "Gradient SPECIFIC kernel test" << std::endl;
    std::cout << GradientS<dim>()(params_s).transpose() << std::endl;

    utils_cpp::DerivativeChecker checker;

    std::cout << "Checking the gradient TOTAL kernel..." << std::endl;
    if (checker.setDimension(k.sizeParams()).checkGradient(FunctionT<dim>(), GradientT<dim>()))
        std::cout << "The gradient is CORRECT!" << std::endl;
    else
        std::cout << "The gradient is NOT correct!" << std::endl;

    std::cout << "Checking the gradient SPECIFIC kernel..." << std::endl;
    if (checker.setDimension(k.sizeParams() - 2).checkGradient(FunctionS<dim>(), GradientS<dim>()))
        std::cout << "The gradient is CORRECT!" << std::endl;
    else
        std::cout << "The gradient is NOT correct!" << std::endl;

    return 0;
}
