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

#include <utils_lib/Timer.hpp>

#include <kernel_lib/kernels/RiemannMatern.hpp>
#include <kernel_lib/kernels/RiemannSqExp.hpp>
#include <kernel_lib/kernels/SquaredExp.hpp>
#include <kernel_lib/utils/Expansion.hpp>

// Kernel
#define EXPANSION utils::Expansion<ParamsEigenfunction, kernels::SquaredExp<ParamsEigenfunction>>
#define RIEMANNSQUAREDEXP kernels::RiemannSqExp<ParamsKernel, EXPANSION>
#define RIEMANNMATERN kernels::RiemannMatern<ParamsKernel, EXPANSION>

#define KERNEL RIEMANNMATERN

using namespace utils_lib;
using namespace kernel_lib;

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

int main(int argc, char const* argv[])
{
    constexpr int dim = 3, num_samples = 100, num_nodes = 100, num_modes = 10;

    // Data
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, dim), Y = Eigen::MatrixXd::Random(num_samples, dim);

    // Nodes, eigenvalues and eigenvectors
    Eigen::MatrixXd N = Eigen::MatrixXd::Random(num_nodes, dim);
    Eigen::VectorXd D = Eigen::VectorXd::Random(num_modes);
    Eigen::MatrixXd U = Eigen::MatrixXd::Random(num_nodes, num_modes);

    // Kernel
    KERNEL k;
    for (size_t i = 0; i < 10; i++) {
        EXPANSION f; // Create eigenfunction
        f.setSamples(N).setWeights(U.col(i)); // Set manifold sampled points and weights
        k.addPair(D(i), f); // Add eigen-pair to Riemann kernel
    }

    std::cout << "BENCHMARK GRAM" << std::endl;
    {
        Timer timer;
        k.gram<dim>(X, Y);
    }

    return 0;
}