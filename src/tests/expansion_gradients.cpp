#include <iostream>
#include <kernel_lib/Kernel.hpp>
#include <utils_cpp/UtilsCpp.hpp>

#define KERNEL kernels::SquaredExp<Params>
#define EXPANSION utils::Expansion<Params, KERNEL>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.405465);
        PARAM_SCALAR(double, sn, 0.693147);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, -0.356675);
    };

    struct exp_sq_full {
        PARAM_VECTOR(double, S, 1, 0.5, 0.5, 1);
    };

    struct psiian {
        PARAM_VECTOR(double, mu, 3.4, 10.1);
    };
};

/* EXPANSION: Function in X */
template <int size>
struct FunctionX : public EXPANSION {
    double operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return EXPANSION()(x);
    }
};

// /* EXPANSION: Gradient in X */
// template <int size>
// struct GradientX : public EXPANSION {
//     Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x) const
//     {
//         return EXPANSION::grad(x);
//     }
// };

// /* EXPANSION: Function (log) in PARAMS */
// template <int size>
// struct FunctionParams : public EXPANSION {
//     Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

//     double operator()(const Eigen::VectorXd& params)
//     {
//         EXPANSION::setParams(params);

//         return EXPANSION::log(x);
//     }
// };

// /* EXPANSION: Gradient in PARAMS */
// template <int size>
// struct GradientParams : public EXPANSION {
//     Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

//     Eigen::VectorXd operator()(const Eigen::VectorXd& params)
//     {
//         EXPANSION::setParams(params);

//         return EXPANSION::gradParams(x);
//     }
// };

int main(int argc, char const* argv[])
{
    constexpr int dim = 2;
    EXPANSION psi;

    // Eigen::VectorXd x = Eigen::VectorXd::Random(dim),
    //                 params = Eigen::VectorXd::Random(psi.sizeParams());

    // std::cout << "EXPANSION: Function in X test" << std::endl;
    // std::cout << FunctionX<dim>()(x) << std::endl;

    // std::cout << "EXPANSION: Gradient in X test" << std::endl;
    // std::cout << GradientX<dim>()(x).transpose() << std::endl;

    // std::cout << "EXPANSION: Function (log) in PARAMS" << std::endl;
    // std::cout << FunctionLogParams<dim>()(params) << std::endl;

    // std::cout << "EXPANSION: Gradient (log) in PARAMS" << std::endl;
    // std::cout << GradientLogParams<dim>()(params).transpose() << std::endl;

    // utils_cpp::DerivativeChecker checker(dim);

    // if (checker.checkGradient(FunctionX<dim>(), GradientX<dim>()))
    //     std::cout << "EXPANSION: The X gradient is CORRECT!" << std::endl;
    // else
    //     std::cout << "EXPANSION: The X gradient is NOT correct!" << std::endl;

    // if (checker.setDimension(psi.sizeParams()).checkGradient(FunctionLogParams<dim>(), GradientLogParams<dim>()))
    //     std::cout << "EXPANSION: PARAMS gradient is CORRECT!" << std::endl;
    // else
    //     std::cout << "EXPANSION: PARAMS gradient is NOT correct!" << std::endl;

    return 0;
}