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

#ifndef KERNELLIB_KERNEL_HPP
#define KERNELLIB_KERNEL_HPP

/* Squared Exponential Kernel Spherical Covariance */
#include "kernel_lib/kernels/SquaredExp.hpp"

/* Squared Exponential Kernel Full Covariance */
#include "kernel_lib/kernels/SquaredExpFull.hpp"

/* Riemannian Squared Exponential Kernel */
#include <kernel_lib/kernels/RiemannSqExp.hpp>

/* Riemannian Matern Kernel */
// #include <kernel_lib/kernels/RiemannMatern.hpp>

/* Kernel expansion */
#include "kernel_lib/utils/Expansion.hpp"

/* Build Graph */
#include "kernel_lib/utils/Graph.hpp"

/* Gaussian distribution */
#include "kernel_lib/utils/Gaussian.hpp"

/* Tools */
#include "kernel_lib/tools/helper.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {
    /* Generic arameters */
    struct ParamsDefaults {
        struct kernel : public defaults::kernel {
        };
        struct exp_sq : public defaults::exp_sq {
        };
        struct exp_sq_full : public defaults::exp_sq_full {
        };
        struct riemann_exp_sq : public defaults::riemann_exp_sq {
        };
        // struct riemann_matern : public defaults::riemann_matern {
        // };
        struct expansion : public defaults::expansion {
        };
        struct gaussian : public defaults::gaussian {
        };
    };

    /** Squared Exponential Spherical Kernel and Expansion
     * @default parameters defintion for kernel and expansion
     */
    using SqExp = kernels::SquaredExp<ParamsDefaults>; // typedef kernel::Rbf<ParamsRbfSpherical> RbfSpherical;
    using SumSqExp = utils::Expansion<ParamsDefaults, SqExp>; // template <typename Params> using SumRbf = Expansion<Params, RbfSpherical>;
} // namespace kernel_lib

#endif // KERNELLIB_KERNEL_HPP