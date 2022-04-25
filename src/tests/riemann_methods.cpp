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

#include <utils_lib/FileManager.hpp>

#include <kernel_lib/kernels/RiemannMatern.hpp>
#include <kernel_lib/kernels/RiemannSqExp.hpp>
#include <kernel_lib/kernels/SquaredExp.hpp>

using namespace kernel_lib;
using namespace utils_lib;

struct ParamsKernel {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.5);
        PARAM_SCALAR(double, sn, 1.4);
    };

    struct riemann_exp_sq : public defaults::riemann_exp_sq {
        PARAM_SCALAR(double, l, 1);
    };

    struct riemann_matern : public defaults::riemann_matern {
        PARAM_SCALAR(double, l, 1);

        PARAM_SCALAR(double, d, 2);

        PARAM_SCALAR(double, nu, 1.5);
    };
};

struct ParamsEigenfunction {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0);
        PARAM_SCALAR(double, sn, -5);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, -2.30259);
    };
};

// Kernel
using Expansion_t = utils::Expansion<ParamsEigenfunction, kernels::SquaredExp<ParamsEigenfunction>>;
#define RIEMANNSQUAREDEXP kernels::RiemannSqExp<ParamsKernel, Expansion_t>
#define RIEMANNMATERN kernels::RiemannMatern<ParamsKernel, Expansion_t>
#define KERNEL RIEMANNMATERN

int main(int argc, char const* argv[])
{
    // Data
    FileManager mn;
    Eigen::MatrixXd X = mn.setFile("rsc/x_samples.csv").read<Eigen::MatrixXd>(),
                    Y = mn.setFile("rsc/y_samples.csv").read<Eigen::MatrixXd>();

    Eigen::VectorXd x = X.row(0),
                    y = Y.row(0);

    // Kernel
    std::cout << "KERNEL CREATION AND PARAMS SIZE" << std::endl;
    using Kernel_t = KERNEL;
    Kernel_t k;

    // Samples, eigenvalues and eigenvectors
    Eigen::MatrixXd N = mn.setFile("rsc/nodes.csv").read<Eigen::MatrixXd>();
    Eigen::VectorXd D = mn.setFile("rsc/eigval.csv").read<Eigen::MatrixXd>();
    Eigen::MatrixXd U = mn.setFile("rsc/eigvec.csv").read<Eigen::MatrixXd>().transpose();

    for (size_t i = 0; i < 10; i++) {
        utils::Expansion<ParamsEigenfunction, kernels::SquaredExp<ParamsEigenfunction>> f; // Create eigenfunction
        f.setSamples(N).setWeights(U.col(i)); // Set manifold sampled points and weights
        k.addPair(D(i), f); // Add eigen-pair to Riemann kernel
    }

    Eigen::VectorXd params(3);
    params << std::log(1.5), std::log(2.0), std::log(0.7);

    std::cout << "DEFAULT PARAMS" << std::endl;
    std::cout << k.sizeParams() << " - " << k.params().transpose() << std::endl;

    std::cout << "SET PARAMS" << std::endl;
    std::cout << params.transpose() << std::endl;
    k.setParams(params);
    std::cout << k.params().transpose() << std::endl;

    std::cout << "KERNEL SINGLE POINTS xy, Xy, XX" << std::endl;
    std::cout << k(x, y) << " - " << k(X.row(0), y) << " - " << k(X.row(0), X.row(0)) << std::endl;

    std::cout << "KERNEL GRAM XY" << std::endl;
    std::cout << k.gram(X, Y) << std::endl;

    std::cout << "KERNEL GRAM XX" << std::endl;
    std::cout << k.gram(X, X) << std::endl;

    std::cout << "GRADIENT SINGLE POINTS xy, Xy, XX" << std::endl;
    std::cout << k.grad(x, y).transpose() << " - " << k.grad(X.row(0), y).transpose() << " - " << k.grad(X.row(0), X.row(0)).transpose() << std::endl;

    std::cout << "GRADIENT XY" << std::endl;
    std::cout << k.gramGrad(X, Y) << std::endl;

    std::cout << "GRADIENT XX" << std::endl;
    std::cout << k.gramGrad(X, X) << std::endl;

    // std::cout << "HESSIAN SINGLE POINTS xy, Xy, XX" << std::endl;
    // std::cout << k.hess(x, y) << std::endl;
    // std::cout << " - " << std::endl;
    // std::cout << k.hess(X.row(0), y) << std::endl;
    // std::cout << " - " << std::endl;
    // std::cout << k.hess(X.row(0), X.row(0)) << std::endl;

    // std::cout << "HESSIAN XY" << std::endl;
    // std::cout << k.gramHess(X, Y) << std::endl;

    // std::cout << "HESSIAN XX" << std::endl;
    // std::cout << k.gramHess(X, X) << std::endl;

    std::cout << "GRADIENT PARAMS SINGLE POINTS xy, Xy, XX" << std::endl;
    std::cout << k.gradParams(x, y).transpose() << " - " << k.gradParams(X.row(0), y).transpose() << " - " << k.gradParams(X.row(0), X.row(0)).transpose() << std::endl;

    std::cout << "GRADIENT PARAMS XY" << std::endl;
    std::cout << k.gramGradParams(X, Y) << std::endl;

    std::cout << "GRADIENT PARAMS XX" << std::endl;
    std::cout << k.gramGradParams(X, X) << std::endl;

    return 0;
}