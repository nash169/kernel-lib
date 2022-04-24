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

#include <kernel_lib/Kernel.hpp>

#include <utils_lib/DerivativeChecker.hpp>
#include <utils_lib/FileManager.hpp>

using namespace utils_lib;
using namespace kernel_lib;

#define KERNEL kernels::SquaredExp<Params>
#define EXPANSION utils::Expansion<Params, KERNEL>

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
struct Function {
    Function(EXPANSION& f) : _f(f) {}

    double operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return _f(x);
    }

    EXPANSION& _f;
};

/* EXPANSION: Gradient in X */
template <int size>
struct Gradient {
    Gradient(EXPANSION& f) : _f(f) {}

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return _f.grad(x);
    }

    EXPANSION& _f;
};

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
    // Data
    FileManager mn;
    Eigen::MatrixXd X = mn.setFile("rsc/y_samples.csv").read<Eigen::MatrixXd>();
    Eigen::VectorXd w = Eigen::VectorXd::Random(X.rows()),
                    x = Eigen::VectorXd::Random(3);

    constexpr int dim = 3;
    EXPANSION psi;
    psi.setSamples(X).setWeights(w);

    std::cout << "EXPANSION: Function in test" << std::endl;
    Function<dim> f(psi);
    std::cout << f(x) << std::endl;

    std::cout << "EXPANSION: Gradient in test" << std::endl;
    Gradient<dim> g(psi);
    std::cout << g(x).transpose() << std::endl;

    // std::cout << "EXPANSION: Function (log) in PARAMS" << std::endl;
    // std::cout << FunctionLogParams<dim>()(params) << std::endl;

    // std::cout << "EXPANSION: Gradient (log) in PARAMS" << std::endl;
    // std::cout << GradientLogParams<dim>()(params).transpose() << std::endl;

    DerivativeChecker checker(dim);

    if (checker.checkGradient(f, g))
        std::cout << "EXPANSION: The gradient is CORRECT!" << std::endl;
    else
        std::cout << "EXPANSION: The gradient is NOT correct!" << std::endl;

    // if (checker.setDimension(psi.sizeParams()).checkGradient(FunctionLogParams<dim>(), GradientLogParams<dim>()))
    //     std::cout << "EXPANSION: PARAMS gradient is CORRECT!" << std::endl;
    // else
    //     std::cout << "EXPANSION: PARAMS gradient is NOT correct!" << std::endl;

    return 0;
}