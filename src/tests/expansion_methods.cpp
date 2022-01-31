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

using namespace kernel_lib;

#define KERNEL kernels::SquaredExp<Params>

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.5);
        PARAM_SCALAR(double, sn, 1.4);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, 3.1);
    };
};

int main(int argc, char const* argv[])
{
    // Data
    Eigen::MatrixXd x(3, 2), x_i(5, 2);
    x << 0.097540404999410, 0.957506835434298,
        0.278498218867048, 0.964888535199277,
        0.546881519204984, 0.157613081677548;

    x_i << 0.970592781760616, 0.141886338627215,
        0.957166948242946, 0.421761282626275,
        0.485375648722841, 0.915735525189067,
        0.800280468888800, 0.792207329559554,
        0.097540404999410, 0.278498218867048;

    Eigen::VectorXd a(2);
    a << 0.097540404999410, 0.957506835434298;

    // Params
    Eigen::VectorXd params(3);
    params << std::log(1.5), std::log(2.0), std::log(0.7);

    // Weights
    Eigen::VectorXd weights(x_i.rows());
    weights << 0.823457828327293, 0.694828622975817, 0.317099480060861, 0.950222048838355, 0.034446080502909;

    std::cout << "EXPANSION CREATION";
    using Kernel_t = KERNEL;
    using Expansion_t = utils::Expansion<Params, Kernel_t>;
    Expansion_t f;
    f.kernel().setParams(params);
    std::cout << "DONE";

    std::cout << "EXPANSION PARAMS" << std::endl;
    f.setWeights(weights);
    std::cout << f.weights().transpose() << std::endl;

    std::cout << "EXPANSION REFERENCE" << std::endl;
    f.setSamples(x_i);
    std::cout << f.samples().transpose() << std::endl;

    std::cout << "EXPANSION SINGLE POINT" << std::endl;
    std::cout << f(a) << std::endl;

    std::cout << "EXPANSION MULTIPLE POINTS" << std::endl;
    std::cout << f.multiEval(x).transpose() << std::endl;

    std::cout << "EXPANSION GRADIENT" << std::endl;
    std::cout << f.grad(a).transpose() << std::endl;

    std::cout << "EXPANSION MULTIPLE GRADIENTS" << std::endl;
    std::cout << f.multiGrad(x) << std::endl;

    std::cout << "EXPANSION HESSIAN" << std::endl;
    std::cout << f.hess(a) << std::endl;

    std::cout << "EXPANSION MULTIPLE HESSIANS" << std::endl;
    std::cout << f.multiHess(x) << std::endl;

    return 0;
}