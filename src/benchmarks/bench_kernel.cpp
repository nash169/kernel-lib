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

#include <kernel_lib/kernels/SquaredExp.hpp>
#include <kernel_lib/kernels/SquaredExpFull.hpp>

#define SQUAREDEXP kernels::SquaredExp<Params>
#define SQUAREDEXPFULL kernels::SquaredExpFull<Params>

#define KERNEL SQUAREDEXP
#define GRAM

using namespace utils_lib;
using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.405465);
        PARAM_SCALAR(double, sn, 0.693147);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, -0.356675);
    };
};

int main(int argc, char const* argv[])
{
    constexpr int dim = 2, num_samples = 20000;

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, dim), Y = Eigen::MatrixXd::Random(num_samples, dim);
    KERNEL k;

    std::cout << "BENCHMARK GRAM" << std::endl;
    {
        Timer timer;
        k.gram<dim>(X, Y);
    }

    return 0;
}