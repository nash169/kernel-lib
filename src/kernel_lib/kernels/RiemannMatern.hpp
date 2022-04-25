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

#ifndef KERNELLIB_KERNELS_RIEMANNMATERN_HPP
#define KERNELLIB_KERNELS_RIEMANNMATERN_HPP

#include "kernel_lib/kernels/RiemannSqExp.hpp"

namespace kernel_lib {

    namespace defaults {
        struct riemann_matern {
            // Log length
            PARAM_SCALAR(double, l, 1);

            // Topological dimension
            PARAM_SCALAR(double, d, 2);

            // Smoothness parameter
            PARAM_SCALAR(double, nu, 1.5);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params, typename EigenFunction>
        class RiemannMatern : public RiemannSqExp<Params, EigenFunction> {
        public:
            RiemannMatern() : RiemannSqExp<Params, EigenFunction>(std::exp(Params::riemann_matern::l())),
                              _dim(Params::riemann_matern::d()),
                              _nu(Params::riemann_matern::nu()) {}

        protected:
            // Length scale
            using RiemannSqExp<Params, EigenFunction>::_l;

            // Eigenvalues
            using RiemannSqExp<Params, EigenFunction>::_d;

            // Eigenfunctions
            using RiemannSqExp<Params, EigenFunction>::_f;

            // Spectral density
            using RiemannSqExp<Params, EigenFunction>::_s;

            double _nu, _dim;

            void spectral()
            {
                _s = (1 / std::pow(_l, 2) + _d.array()).pow(-_nu - 0.5 * _dim);
                _s /= _s.sum();
            }

            Eigen::VectorXd spectralGrad() const
            {
                double a = -_nu - 0.5 * _dim;
                Eigen::VectorXd ds = 2 * a * std::pow(_l, -2) * (std::pow(_l, -2) + _d.array()).pow(-1);
                return _s.array() * ((ds.array() * _s.array()).sum() - ds.array());
            }
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_RIEMANNMATERN_HPP