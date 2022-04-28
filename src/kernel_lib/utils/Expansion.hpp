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
#ifndef KERNELLIB_EXPANSION_UTILS_HPP
#define KERNELLIB_EXPANSION_UTILS_HPP

#include "kernel_lib/tools/type_name_rt.hpp"

namespace kernel_lib {
    namespace defaults {
        struct expansion {
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class SquaredExp;

        template <typename Params, typename Function>
        class RiemannSqExp;
    } // namespace kernels

    namespace utils {
        template <typename Params, typename Kernel>
        class Expansion {
        public:
            Expansion() : _k() {}

            template <int Size>
            EIGEN_ALWAYS_INLINE double operator()(const Eigen::Matrix<double, Size, 1>& x) const
            {
                double r = 0;

                // Can I parallelize this?
                // https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-data.html
                // https://stackoverflow.com/questions/11773115/parallel-for-loop-in-openmp
                // https://stackoverflow.com/questions/40495250/openmp-reduction-with-eigenvectorxd

                // #pragma omp parallel for reduction(+ \
//                                    : r)
                for (size_t i = 0; i < _x.rows(); i++)
                    r += _w(i) * _k.template kernelImpl<Size>(_x.row(i), x);

                return r;
            }

            template <int Size>
            Eigen::VectorXd multiEval(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x) const
            {
                Eigen::VectorXd r(x.rows());

#pragma omp parallel for
                for (size_t i = 0; i < r.rows(); i++)
                    r(i) = this->operator()<Size>(x.row(i));

                return r;
            }

            template <int Size>
            Eigen::VectorXd multiEval2(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x) const
            {
                return _k.gram(_x, x).transpose() * _w;
            }

            // Check https://stackoverflow.com/questions/47035541/specialize-only-a-part-of-one-method-of-a-template-class
            //       https://stackoverflow.com/questions/12877420/specialization-of-template-class-method
            Eigen::VectorXd temp(const Eigen::MatrixXd& x) const
            {
                if constexpr (std::is_same_v<Kernel, kernels::SquaredExp<Params>>) {
                    std::cout << "Hello" << std::endl;
                }

                // Check https://github.com/willwray/type_name
                std::cout << type_name_str<Kernel>() << std::endl;

                Eigen::VectorXd r(x.rows());
                return r;
            }

            /* Gradient */
            template <int Size>
            EIGEN_ALWAYS_INLINE Eigen::Matrix<double, Size, 1> grad(const Eigen::Matrix<double, Size, 1>& x) const
            {
                Eigen::Matrix<double, Size, 1> grad = Eigen::VectorXd::Zero(x.size());

                for (size_t i = 0; i < _x.rows(); i++)
                    grad += _w(i) * _k.template gradImpl<Size>(_x.row(i), x, 0);

                return grad;
            }

            /* Gradient  multiple points */
            template <int Size>
            Eigen::Matrix<double, Eigen::Dynamic, Size> multiGrad(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x) const
            {
                Eigen::Matrix<double, Eigen::Dynamic, Size> grad(x.rows(), x.cols());

#pragma omp parallel for
                for (size_t i = 0; i < grad.rows(); i++)
                    grad.row(i) = this->grad<Size>(x.row(i));

                return grad;
            }

            /* Hessian */
            template <int Size>
            EIGEN_ALWAYS_INLINE Eigen::Matrix<double, Size, Size> hess(const Eigen::Matrix<double, Size, 1>& x) const
            {
                Eigen::Matrix<double, Size, Size> H = Eigen::MatrixXd::Zero(x.rows(), x.rows());

                for (size_t i = 0; i < _x.rows(); i++)
                    H += _w(i) * _k.template hessImpl<Size>(_x.row(i), x);

                return H;
            }

            /* Hessian  multiple points */
            template <int Size>
            Eigen::MatrixXd multiHess(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x) const
            {
                Eigen::MatrixXd H(x.rows(), x.cols() * x.cols());

#pragma omp parallel for
                for (size_t i = 0; i < H.rows(); i++)
                    H.row(i) = Eigen::Map<Eigen::VectorXd>(this->hess<Size>(x.row(i)).data(), x.cols() * x.cols());

                return H;
            }

            const Eigen::MatrixXd& samples() const { return _x; }

            const Eigen::VectorXd& weights() const { return _w; }

            Kernel& kernel() { return _k; }

            virtual Expansion& setSamples(const Eigen::MatrixXd& x)
            {
                _x = x;

                return *this;
            }

            virtual Expansion& setWeights(const Eigen::VectorXd& w)
            {
                _w = w;

                return *this;
            }

        protected:
            /* Kernel */
            Kernel _k;

            /* Parameters */
            Eigen::VectorXd _w;

            /* Referece points */
            Eigen::MatrixXd _x;
        };

        // template <>
        // Eigen::VectorXd Expansion<float, kernels::SquaredExp<float>>::temp(const Eigen::MatrixXd& x) const
        // {
        //     Eigen::VectorXd r(x.rows());

        //     return r;
        // }
    } // namespace utils
} // namespace kernel_lib

#endif // KERNELLIB_EXPANSION_UTILS_HPP