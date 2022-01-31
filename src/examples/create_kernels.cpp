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

#include <utils_lib/FileManager.hpp>

using namespace utils_lib;
using namespace kernel_lib;

int main(int argc, char const* argv[])
{
    double box[] = {0, 100, 0, 100};
    size_t resolution = 100, num_samples = resolution * resolution, dim = sizeof(box) / sizeof(*box) / 2;

    // Data
    Eigen::MatrixXd X = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    Y = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    x_test(num_samples, dim), x_train(1, dim);
    x_test << Eigen::Map<Eigen::VectorXd>(X.data(), X.size()), Eigen::Map<Eigen::VectorXd>(Y.data(), Y.size());
    x_train << 50, 50;

    // Reference in the shape of the vector does not work for the moment
    // Eigen::Vector2d x_train = (Eigen::Vector2d() << 50, 50).finished();

    // Kernel
    using Kernel_t = kernels::SquaredExp<ParamsDefaults>;
    Kernel_t k;

    // Expansion
    using Expansion_t = utils::Expansion<ParamsDefaults, Kernel_t>;
    Expansion_t psi;
    psi.setSamples(x_train);
    psi.setWeights(tools::makeVector(1));

    // Solution
    FileManager io_manager;
    io_manager.setFile("rsc/kernel_eval.csv");
    io_manager.write("X", X, "Y", Y, "F", psi.multiEval(x_test));

    return 0;
}