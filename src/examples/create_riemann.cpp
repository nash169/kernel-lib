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

#include <kernel_lib/Kernel.hpp>
#include <utils_lib/FileManager.hpp>

using namespace utils_lib;
using namespace kernel_lib;

struct ParamsExp {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0);
        PARAM_SCALAR(double, sn, -5);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, -2.30259);
    };
};

struct ParamsRiemann {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0);
        PARAM_SCALAR(double, sn, -5);
    };

    struct riemann_exp_sq : public defaults::riemann_exp_sq {
        PARAM_SCALAR(double, l, 1);
    };
};

int main(int argc, char const* argv[])
{
    std::string manifold = (argc > 1) ? argv[1] : "torus";
    FileManager mn;

    // Samples on manifold
    Eigen::MatrixXd X = mn.setFile("rsc/" + manifold + "_nodes.csv").read<Eigen::MatrixXd>(),
                    I = mn.setFile("rsc/" + manifold + "_elems.csv").read<Eigen::MatrixXd>();

    // Eigenvalues and eigenvectors
    Eigen::VectorXd D = mn.setFile("rsc/" + manifold + "_eigval.csv").read<Eigen::MatrixXd>();
    Eigen::MatrixXd U = mn.setFile("rsc/" + manifold + "_eigvec.csv").read<Eigen::MatrixXd>().transpose();

    // Kernel
    using Kernel_t = kernels::SquaredExp<ParamsExp>;
    using Expansion_t = utils::Expansion<ParamsExp, Kernel_t>;
    using Riemann_t = kernels::RiemannSqExp<ParamsRiemann, Expansion_t>;
    Riemann_t k;

    int num_modes = (argc > 2) ? std::stoi(argv[2]) : 10;

    for (size_t i = 0; i < num_modes; i++) {
        Expansion_t f; // Create eigenfunction
        f.setSamples(X).setWeights(U.col(i)); // Set manifold sampled points and weights
        k.addPair(D(i), f); // Add eigen-pair to Riemann kernel
    }

    // Space
    std::random_device rd;
    std::default_random_engine eng(rd());
    constexpr int RAND_NUMS_TO_GENERATE = 10;
    constexpr size_t resolution = 100, num_samples = resolution * resolution, dim = 2;
    Eigen::MatrixXd X_chart(num_samples, dim), X_embed(num_samples, dim + 1), X_train(RAND_NUMS_TO_GENERATE, dim + 1);

    if (!manifold.compare("sphere")) {
        Eigen::MatrixXd x = Eigen::RowVectorXd::LinSpaced(resolution, 0, M_PI).replicate(resolution, 1),
                        y = Eigen::VectorXd::LinSpaced(resolution, 0, 2 * M_PI).replicate(1, resolution);
        X_chart << Eigen::Map<Eigen::VectorXd>(x.data(), x.size()), Eigen::Map<Eigen::VectorXd>(y.data(), y.size());

        auto sphere_embed = [](const Eigen::MatrixXd& x) {
            Eigen::MatrixXd e(x.rows(), 3);
            e.col(0) = x.col(0).array().sin() * x.col(1).array().cos();
            e.col(1) = x.col(0).array().sin() * x.col(1).array().sin();
            e.col(2) = x.col(0).array().cos();
            return e;
        };
        X_embed = sphere_embed(X_chart);

        std::uniform_real_distribution<double> x_distr(0, M_PI), y_distr(0, 2 * M_PI);
        for (size_t i = 0; i < RAND_NUMS_TO_GENERATE; i++)
            X_train.row(i) = sphere_embed(Eigen::RowVector2d(x_distr(eng), y_distr(eng)));
    }
    else if (!manifold.compare("torus")) {
        Eigen::MatrixXd x = Eigen::RowVectorXd::LinSpaced(resolution, 0, 2 * M_PI).replicate(resolution, 1),
                        y = Eigen::VectorXd::LinSpaced(resolution, 0, 2 * M_PI).replicate(1, resolution);
        X_chart << Eigen::Map<Eigen::VectorXd>(x.data(), x.size()), Eigen::Map<Eigen::VectorXd>(y.data(), y.size());

        auto torus_embed = [](const Eigen::MatrixXd& x) {
            double a = 1, b = 3;
            Eigen::MatrixXd e(x.rows(), 3);
            e.col(0) = (b + a * x.col(0).array().cos()) * x.col(1).array().cos();
            e.col(1) = (b + a * x.col(0).array().cos()) * x.col(1).array().sin();
            e.col(2) = a * x.col(0).array().sin();
            return e;
        };
        X_embed = torus_embed(X_chart);

        std::uniform_real_distribution<double> x_distr(0, 2 * M_PI), y_distr(0, 2 * M_PI);
        for (size_t i = 0; i < RAND_NUMS_TO_GENERATE; i++)
            X_train.row(i) = torus_embed(Eigen::RowVector2d(x_distr(eng), y_distr(eng)));
    }

    mn.setFile("rsc/riemann.csv")
        .write("NODES", X, "CHART", X_chart, "EMBED", X_embed, "INDEX", I,
            "SURF", k.gram(X_embed, Eigen::MatrixXd(X.row(100))),
            "MESH", k.gram(X, Eigen::MatrixXd(X.row(100))),
            "GRAM", k.gram(X_train, X_train));

    return 0;
}