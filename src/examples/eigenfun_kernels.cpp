/*
    This file is part of kernel-lib.

    Copyright (c) 2020, 2021, 2022 Bernardo Fichera <bernardo.fichera@gmail.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <iostream>
#include <random>
#include <vector>

#include <kernel_lib/kernels/RiemannSqExp.hpp>
#include <kernel_lib/kernels/SquaredExp.hpp>
#include <kernel_lib/utils/Expansion.hpp>

#include <utils_lib/DerivativeChecker.hpp>

using namespace utils_lib;
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

template <int size, typename Kernel>
struct FunctionParamsT {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    FunctionParamsT(Kernel ker) : k(ker) {}

    double operator()(const Eigen::VectorXd& params)
    {
        k.setParams(params);

        return k(x, y);
    }

    Kernel k;
};

template <int size, typename Kernel>
struct GradientParamsT {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    GradientParamsT(Kernel ker) : k(ker) {}

    Eigen::VectorXd operator()(const Eigen::VectorXd& params)
    {
        k.setParams(params);

        return k.gradParams(x, y);
    }

    Kernel k;
};

int main(int argc, char const* argv[])
{
    // Data
    constexpr int dim = 2, num_X = 1000, num_Y = 500, num_modes = 50, num_reference = 10000;

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_X, dim), Y = Eigen::MatrixXd::Random(num_Y, dim);
    Eigen::VectorXd a = Eigen::VectorXd::Random(dim), b = Eigen::VectorXd::Random(dim), W = Eigen::VectorXd::Random(num_X);

    // Data
    Eigen::MatrixXd test_x(3, 2), test_y(5, 2);
    test_x << 0.097540404999410, 0.957506835434298,
        0.278498218867048, 0.964888535199277,
        0.546881519204984, 0.157613081677548;

    test_y << 0.970592781760616, 0.141886338627215,
        0.957166948242946, 0.421761282626275,
        0.485375648722841, 0.915735525189067,
        0.800280468888800, 0.792207329559554,
        0.097540404999410, 0.278498218867048;

    Eigen::VectorXd test_a(2), test_b(2);
    test_a << 0.097540404999410, 0.957506835434298;
    test_b << 0.970592781760616, 0.141886338627215;

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
        f.setSamples(Eigen::MatrixXd::Random(num_reference, dim)).setWeights(Eigen::VectorXd::Random(num_reference));

        // Set pair
        k.addPair(randDouble(), f);
    }

    FunctionParamsT<dim, Riemann_t> function(k);
    GradientParamsT<dim, Riemann_t> gradient(k);

    Eigen::Vector3d params = Eigen::Vector3d::Random();

    // std::cout << function(params) << std::endl;
    // std::cout << gradient(params).transpose() << std::endl;

    DerivativeChecker checker(k.sizeParams());

    std::cout << k.gradParams(test_a, test_b).transpose() << std::endl;
    // checker.checkGradient(function, gradient);
    std::cout << k.gramGradParams(test_x, test_y) << std::endl;

    // Riemann kernel expansion
    using RiemExpansion_t = utils::Expansion<ParamsRiemann, Riemann_t>;
    RiemExpansion_t psi;

    psi.setSamples(X).setWeights(W);

    for (size_t i = 0; i < num_modes; i++) {
        // Create function
        Expansion_t f;
        f.setSamples(Eigen::MatrixXd::Random(num_reference, dim)).setWeights(Eigen::VectorXd::Random(num_reference));

        // Set pair
        psi.kernel().addPair(randDouble(), f);
    }

    // psi.temp(Y);

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
