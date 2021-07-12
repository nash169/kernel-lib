#include <iostream>
#include <random>

#include <kernel_lib/kernels/RiemannSqExp.hpp>
#include <kernel_lib/kernels/SquaredExp.hpp>
#include <kernel_lib/utils/Expansion.hpp>

#include <utils_cpp/UtilsCpp.hpp>

using namespace kernel_lib;

// Kernel parameters
struct ParamsExp {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.405465);
        PARAM_SCALAR(double, sn, 0.693147);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, -0.356675);
    };
};

struct ParamsRiemann {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.405465);
        PARAM_SCALAR(double, sn, 0.693147);
    };

    struct riemann_exp_sq : public defaults::riemann_exp_sq {
        PARAM_SCALAR(double, l, -0.356675);
    };
};

// Example of eigen-function
class SinFunction {
public:
    SinFunction(double h = 1.0) : _h(h) {}

    /* overload the () operator */
    double operator()(const Eigen::VectorXd& x) const
    {
        return eval(x);
    }

    void setFrequency(double h)
    {
        _h = h;
    }

    double eval(const Eigen::VectorXd& x) const
    {
        return std::sin(_h * x.squaredNorm());
    }

protected:
    double _h;
};

// Generate random double
double randDouble(double fMin = 0, double fMax = 1)
{
    double f = (double)std::rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main(int argc, char const* argv[])
{
    // Data
    constexpr int dim = 2, num_X = 1000, num_Y = 500, num_modes = 50, num_reference = 10000;

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_X, dim), Y = Eigen::MatrixXd::Random(num_Y, dim);
    Eigen::VectorXd a = Eigen::VectorXd::Random(dim), b = Eigen::VectorXd::Random(dim), W = Eigen::VectorXd::Random(num_X);

    // Kernel
    using Kernel_t = kernels::SquaredExp<ParamsExp>;
    using Expansion_t = utils::Expansion<ParamsExp, Kernel_t>;
    using Riemann_t = kernels::RiemannSqExp<ParamsRiemann, Expansion_t>;
    Riemann_t k;

    // Set eigenfunctions
    std::srand((unsigned)time(0));

    for (size_t i = 0; i < num_modes; i++) {
        // Create function
        Expansion_t f;
        f.setReference(Eigen::MatrixXd::Random(num_reference, dim)).setParams(Eigen::VectorXd::Random(num_reference));

        // Set pair
        k.addPair(randDouble(), f);
    }

    // std::cout << k(a, b) << std::endl;
    // std::cout << std::endl;
    // std::cout << k.kernel(a, b) << std::endl;
    // std::cout << std::endl;
    // std::cout << k.gram(X, Y) << std::endl;
    // std::cout << std::endl;
    // std::cout << k.gram2(X, Y) << std::endl;
    // std::cout << std::endl;
    // std::cout << k.gram3(X, Y) << std::endl;

    // std::cout << Eigen::nbThreads() << std::endl;
    // {
    //     utils_cpp::Timer timer;
    //     k.gram(X, Y);
    // }

    using RiemExpansion_t = utils::Expansion<ParamsRiemann, Riemann_t>;
    RiemExpansion_t psi;

    psi.setReference(X).setParams(W);

    for (size_t i = 0; i < num_modes; i++) {
        // Create function
        Expansion_t f;
        f.setReference(Eigen::MatrixXd::Random(num_reference, dim)).setParams(Eigen::VectorXd::Random(num_reference));

        // Set pair
        psi.kernel().addPair(randDouble(), f);
    }

    psi.temp(Y);

    // {
    //     utils_cpp::Timer timer;
    //     psi.temp(Y);
    // }

    return 0;
}

// #include <kernel_lib/utils/EigenFunction.hpp>
// // Set of eigen-functions to pass to the kernel
// class MyFunction : public utils::EigenFunction<SinFunction> {
// public:
//     MyFunction()
//     {
//         // // Add 3 eigen-functions of type SinFunction
//         // utils::EigenFunction<size, SinFunction>::addEigenFunctions(SinFunction(0.3), SinFunction(1.2), SinFunction(0.7));

//         // // Add 3 random eigen-values
//         // utils::EigenFunction<size, SinFunction>::addEigenValues(0.5, 0.2, 0.8);

//         // Add unordered map
//         utils::EigenFunction<SinFunction>::addEigenPair(0.5, SinFunction(0.3), 0.2, SinFunction(1.2), 0.8, SinFunction(0.7));
//     }

//     // // Override operator()
//     // inline double operator()(const Eigen::Matrix<double, size, 1>& x, const size_t& i) const override
//     // {
//     //     return utils::EigenFunction<size, SinFunction>::_f[i].eval(x);
//     // }

//     // Override operator()
//     inline double operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1>& x, const double& eigenvalue) const override
//     {
//         return utils::EigenFunction<SinFunction>::_eigen_pair.at(eigenvalue).eval(x);
//     }

// protected:
// };
