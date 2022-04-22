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

struct Params {
    struct kernel : public defaults::kernel {
    };

    struct exp_sq {
        PARAM_SCALAR(double, l, 2);
    };
};

int main(int argc, char const* argv[])
{
    // Kernel
    using Kernel_t = kernels::SquaredExp<Params>;
    Kernel_t k;

    // Space
    constexpr double box[] = {0, 100, 0, 100};
    constexpr size_t resolution = 100,
                     num_samples = resolution * resolution,
                     dim = sizeof(box) / sizeof(*box) / 2;

    // Reference point
    Eigen::MatrixXd x_ref(1, 2);
    x_ref << 50, 50;

    // Test points
    Eigen::MatrixXd X = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    Y = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    X_test(num_samples, dim);
    X_test << Eigen::Map<Eigen::VectorXd>(X.data(), X.size()), Eigen::Map<Eigen::VectorXd>(Y.data(), Y.size());

    // Training points
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> x_distr(box[0], box[1]),
        y_distr(box[2], box[3]);

    constexpr int RAND_NUMS_TO_GENERATE = 10;
    Eigen::MatrixXd X_train(RAND_NUMS_TO_GENERATE, dim);
    for (size_t i = 0; i < RAND_NUMS_TO_GENERATE; i++)
        X_train.row(i) << x_distr(eng), y_distr(eng);

    // Solution
    FileManager io_manager;
    io_manager
        .setFile("rsc/kernel.csv")
        .write("X", X, "Y", Y, "EVAL", k.gram(X_test, x_ref), "GRAM", k.gram(X_train, X_train));

    return 0;
}

// Reference in the shape of the vector does not work for the moment
// Eigen::Vector2d x_train = (Eigen::Vector2d() << 50, 50).finished();
