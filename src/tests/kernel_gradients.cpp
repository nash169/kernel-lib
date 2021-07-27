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

/* TOTAL KERNEL: Function in X */
template <int size>
struct FunctionX : public KERNEL {
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Zero(size);

    double operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return KERNEL()(x, y);
    }
};

/* TOTAL KERNEL: Gradient in X */
template <int size>
struct GradientX : public KERNEL {
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Zero(size);

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return KERNEL::grad(x, y);
    }
};

/* TOTAL KERNEL: Function in Y */
template <int size>
struct FunctionY : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

    double operator()(const Eigen::Matrix<double, size, 1>& y) const
    {
        return KERNEL()(x, y);
    }
};

/* TOTAL KERNEL: Gradient in Y */
template <int size>
struct GradientY : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& y) const
    {
        return KERNEL::grad(x, y, 0);
    }
};

/* TOTAL KERNEL: Function in PARAMS */
template <int size>
struct FunctionParamsT : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    double operator()(const Eigen::VectorXd& params)
    {
        KERNEL::setParams(params);

        return KERNEL::operator()(x, y);
    }
};

/* TOTAL KERNEL: Gradient in PARAMS */
template <int size>
struct GradientParamsT : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    Eigen::VectorXd operator()(const Eigen::VectorXd& params)
    {
        KERNEL::setParams(params);

        return KERNEL::gradParams(x, y);
    }
};

/* SPECIFIC KERNEL: Function in PARAMS */
template <int size>
struct FunctionParamsS : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    double operator()(const Eigen::VectorXd& params)
    {
        KERNEL::setParameters(params);

        return KERNEL::kernel(x, y);
    }
};

/* SPECIFIC KERNEL: Gradient in PARAMS */
template <int size>
struct GradientParamsS : public KERNEL {
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

    Eigen::VectorXd x = Eigen::VectorXd::Random(dim),
                    params_t = Eigen::VectorXd::Random(k.sizeParams()), params_s = Eigen::VectorXd::Random(k.sizeParams() - 2);

    std::cout << "TOTAL KERNEL: Function in X test" << std::endl;
    std::cout << FunctionX<dim>()(x) << std::endl;

    std::cout << "TOTAL KERNEL: Gradient in X test" << std::endl;
    std::cout << GradientX<dim>()(x).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Function in Y test" << std::endl;
    std::cout << FunctionY<dim>()(x) << std::endl;

    std::cout << "TOTAL KERNEL: Gradient in Y test" << std::endl;
    std::cout << GradientY<dim>()(x).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Function in PARAMS" << std::endl;
    std::cout << FunctionParamsT<dim>()(params_t) << std::endl;

    std::cout << "TOTAL KERNEL:: Gradient in PARAMS" << std::endl;
    std::cout << GradientParamsT<dim>()(params_t).transpose() << std::endl;

    std::cout << "SPECIFIC KERNEL: Function in PARAMS" << std::endl;
    std::cout << FunctionParamsS<dim>()(params_s) << std::endl;

    std::cout << "SPECIFIC KERNEL: Gradient in PARAMS" << std::endl;
    std::cout << GradientParamsS<dim>()(params_s).transpose() << std::endl;

    utils_cpp::DerivativeChecker checker(dim);

    if (checker.checkGradient(FunctionX<dim>(), GradientX<dim>()))
        std::cout << "The X gradient is CORRECT!" << std::endl;
    else
        std::cout << "The X gradient is NOT correct!" << std::endl;

    if (checker.checkGradient(FunctionY<dim>(), GradientY<dim>()))
        std::cout << "The Y gradient is CORRECT!" << std::endl;
    else
        std::cout << "The Y gradient is NOT correct!" << std::endl;

    if (checker.setDimension(k.sizeParams()).checkGradient(FunctionParamsT<dim>(), GradientParamsT<dim>()))
        std::cout << "TOTAL KERNEL: PARAMS gradient is CORRECT!" << std::endl;
    else
        std::cout << "TOTAL KERNEL: PARAMS gradient is NOT correct!" << std::endl;

    if (checker.setDimension(k.sizeParams() - 2).checkGradient(FunctionParamsS<dim>(), GradientParamsS<dim>()))
        std::cout << "SPECIFIC KERNEL: PARAMS gradient is CORRECT!" << std::endl;
    else
        std::cout << "SPECIFIC KERNEL: PARAMS gradient is NOT correct!" << std::endl;

    return 0;
}
