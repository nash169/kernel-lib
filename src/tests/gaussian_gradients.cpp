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

using namespace utils_lib;
using namespace kernel_lib;

#define KERNEL kernels::SquaredExpFull<Params>
#define GAUSS utils::Gaussian<Params, KERNEL>

struct Params {
    struct kernel : public defaults::kernel {
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, -0.356675);
    };

    struct exp_sq_full {
        PARAM_VECTOR(double, S, 1, 0.5, 0.5, 1);
    };

    struct gaussian {
        PARAM_VECTOR(double, mu, 0.4, 0.7);
    };
};

/* GAUSSIAN: Function in X */
template <int size>
struct FunctionX : public GAUSS {
    double operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return GAUSS()(x);
    }
};

/* GAUSSIAN: Gradient in X */
template <int size>
struct GradientX : public GAUSS {
    auto operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return GAUSS::grad(x);
    }
};

/* GAUSSIAN: Function (log) in PARAMS */
template <int size>
struct FunctionLogParams : public GAUSS {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

    double operator()(const Eigen::VectorXd& params)
    {
        GAUSS::setParams(params);

        return GAUSS::log(x);
    }
};

/* GAUSSIAN: Gradient in PARAMS */
template <int size>
struct GradientLogParams : public GAUSS {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

    Eigen::VectorXd operator()(const Eigen::VectorXd& params)
    {
        GAUSS::setParams(params);

        return GAUSS::gradParams(x);
    }
};

int main(int argc, char const* argv[])
{
    constexpr int dim = 2;
    GAUSS gauss;

    Eigen::VectorXd x = Eigen::VectorXd::Random(dim),
                    params = Eigen::VectorXd::Random(gauss.sizeParams());

    std::cout << "GAUSSIAN: Function in X test" << std::endl;
    std::cout << FunctionX<dim>()(x) << std::endl;

    std::cout << "GAUSSIAN: Gradient in X test" << std::endl;
    std::cout << GradientX<dim>()(x).transpose() << std::endl;

    std::cout << "GAUSSIAN: Function (log) in PARAMS" << std::endl;
    std::cout << FunctionLogParams<dim>()(params) << std::endl;

    std::cout << "GAUSSIAN: Gradient (log) in PARAMS" << std::endl;
    std::cout << GradientLogParams<dim>()(params).transpose() << std::endl;

    DerivativeChecker checker(dim);

    if (checker.checkGradient(FunctionX<dim>(), GradientX<dim>()))
        std::cout << "GAUSSIAN: The X gradient is CORRECT!" << std::endl;
    else
        std::cout << "GAUSSIAN: The X gradient is NOT correct!" << std::endl;

    if constexpr (std::is_same_v<decltype(gauss), utils::Gaussian<Params, kernels::SquaredExpFull<Params>>>) {
        // Generate symmetric matrices for the full squared exponential kernel
        Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim), B = Eigen::MatrixXd::Random(dim, dim),
                        C(dim, dim), D(dim, dim);
        C = 0.5 * (A + A.transpose()) + dim * Eigen::MatrixXd::Identity(dim, dim);
        D = 0.5 * (B + B.transpose()) + dim * Eigen::MatrixXd::Identity(dim, dim);
        params.segment(0, dim * dim) = Eigen::Map<Eigen::VectorXd>(C.data(), C.size());
        Eigen::VectorXd v = Eigen::VectorXd::Random(gauss.sizeParams());
        v.segment(0, dim * dim) = Eigen::Map<Eigen::VectorXd>(D.data(), D.size());

        if (checker.setDimension(gauss.sizeParams()).checkGradient(FunctionLogParams<dim>(), GradientLogParams<dim>(), params, v))
            std::cout << "GAUSSIAN: PARAMS gradient is CORRECT!" << std::endl;
        else
            std::cout << "GAUSSIAN: PARAMS gradient is NOT correct!" << std::endl;
    }
    else {
        if (checker.setDimension(gauss.sizeParams()).checkGradient(FunctionLogParams<dim>(), GradientLogParams<dim>()))
            std::cout << "GAUSSIAN: PARAMS gradient is CORRECT!" << std::endl;
        else
            std::cout << "GAUSSIAN: PARAMS gradient is NOT correct!" << std::endl;
    }

    return 0;
}