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

#ifndef KERNELLIB_KERNELS_RIEMANNSQEXP_HPP
#define KERNELLIB_KERNELS_RIEMANNSQEXP_HPP

#define VECTORIZED_EXPANSION

#include <unordered_map>

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

#include "kernel_lib/tools/helper.hpp"

namespace kernel_lib {

    namespace defaults {
        struct riemann_exp_sq {
            // Log length
            PARAM_SCALAR(double, l, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params, typename Function>
        class RiemannSqExp : public AbstractKernel<Params, RiemannSqExp<Params, Function>> {
        public:
            using KernelFunction = Function;
            RiemannSqExp() : _l(std::exp(Params::riemann_exp_sq::l())) {}

            /* Overload kernel for handling single sample evaluation */
            // template <int Size>
            // EIGEN_ALWAYS_INLINE double kernel(const Eigen::Matrix<double, Size, 1>& x, const Eigen::Matrix<double, Size, 1>& y) const
            // {
            //     double r = 0, n = 0;

            //     // this has to parallelized
            //     for (auto& pair : _pairs) {
            //         n += std::exp(-0.5 * std::pow(_l, 2));
            //         r += std::exp(-0.5 * std::pow(_l, 2) * pair.first) * pair.second(x) * pair.second(y);
            //     }

            //     return r / n;
            // }

            template <typename Derived>
            EIGEN_ALWAYS_INLINE double kernel(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                double r = 0, n = 0;

                // this has to parallelized
                for (auto& pair : _pairs) {
                    n += std::exp(-0.5 * std::pow(_l, 2) * pair.first);
                    r += std::exp(-0.5 * std::pow(_l, 2) * pair.first) * pair.second.template operator() < (Derived::RowsAtCompileTime == 1) ? Derived::ColsAtCompileTime : Derived::RowsAtCompileTime > (x)*pair.second.template operator() < (Derived::RowsAtCompileTime == 1) ? Derived::ColsAtCompileTime
                                                                                                                                                                                                                                                                                 : Derived::RowsAtCompileTime > (y);
                }

                return r / n;
            }

            template <int Size>
            EIGEN_DEVICE_FUNC Eigen::MatrixXd gram(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x, const Eigen::Matrix<double, Eigen::Dynamic, Size>& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows();
                Eigen::MatrixXd k = Eigen::MatrixXd::Zero(x_samples, y_samples);
                double n = 0;

                // #pragma omp parallel
                for (auto& pair : _pairs) {
                    n += std::exp(-0.5 * std::pow(_l, 2) * pair.first);
                    k += std::exp(-0.5 * std::pow(_l, 2) * pair.first) * pair.second.multiEval(x) * pair.second.multiEval(y).transpose();
                }

                //                 for (auto& pair : _pairs) {
                //                     n += std::exp(-0.5 * std::pow(_l, 2) * pair.first);
                //                     Eigen::VectorXd fx = pair.second.multiEval(x), fy = pair.second.multiEval(y);
                // #pragma omp parallel for collapse(2)
                //                     for (size_t j = 0; j < y_samples; j++)
                //                         for (size_t i = 0; i < x_samples; i++)
                //                             k(i, j) += std::exp(-0.5 * std::pow(_l, 2) * pair.first) * fx(i) * fy(j);
                //                 }

                k *= _sf2 / n;

                if (x.data() == y.data())
                    k.diagonal().array() += _sn2;

                return k;
            }

            template <typename Derived>
            EIGEN_ALWAYS_INLINE auto gradient(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, const size_t& i = 1) const
            {
                return Eigen::VectorXd::Zero(x.size());
            }

            template <typename Derived>
            EIGEN_ALWAYS_INLINE double gradientParams(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, const size_t& i = 1) const
            {
                double r = 0, dr = 0, dn = 0, n = 0, cn = 0, cr = 0;

                for (auto& pair : _pairs) {
                    // Repeated term
                    cn = std::exp(-0.5 * std::pow(_l, 2) * pair.first);

                    // Normalization term
                    n += cn;

                    // Normalization derivative
                    dn += pair.first * cn;

                    // Repeated term
                    cr = cn * pair.second.template operator()
                            < (Derived::RowsAtCompileTime == 1)
                        ? Derived::ColsAtCompileTime
                        : Derived::RowsAtCompileTime > (x)*pair.second.template operator() < (Derived::RowsAtCompileTime == 1) ? Derived::ColsAtCompileTime
                                                                                                                               : Derived::RowsAtCompileTime > (y);
                    // Kernel
                    r += cr;

                    // Kernel derivative
                    dr += pair.first * cr;
                }

                return std::pow(_l, 2) / n * (dn * r / n - dr);

                // for (auto& i : _f.eigenPair()) {
                //     n += std::exp(-0.5 * std::pow(_l, 2));
                //     r -= std::pow(_l, 2) * i.first * std::exp(-0.5 * std::pow(_l, 2) * i.first) * _f(x, i.first) * _f(y, i.first);
                // }

                // return r / n / _f.eigenPair().size();
            }

            template <int Size>
            EIGEN_DEVICE_FUNC Eigen::MatrixXd gramGradParams(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x, const Eigen::Matrix<double, Eigen::Dynamic, Size>& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows();

                Eigen::MatrixXd grad(x_samples * y_samples, 3), r(x_samples, y_samples);
                double n = 0, dn = 0, c = 0;

                for (auto& pair : _pairs) {
                    n += std::exp(-0.5 * std::pow(_l, 2) * pair.first);
                    dn += pair.first * std::exp(-0.5 * std::pow(_l, 2) * pair.first);
                }

                // #pragma omp parallel
                for (auto& pair : _pairs) {
                    c = std::exp(-0.5 * std::pow(_l, 2) * pair.first);

                    r = c / n * pair.second.multiEval(x) * pair.second.multiEval(y).transpose();

                    grad.col(0) += 2 * _sf2 * Eigen::Map<Eigen::VectorXd>(r.data(), x_samples * y_samples);

                    grad.col(2) += _sf2 * std::pow(_l, 2) * (dn / n - pair.first) * Eigen::Map<Eigen::VectorXd>(r.data(), x_samples * y_samples);
                }

                if (x.data() == y.data()) {
                    r = 2 * _sn2 * Eigen::MatrixXd::Identity(x_samples, y_samples);
                    grad.col(1) = Eigen::Map<Eigen::VectorXd>(r.data(), x_samples * y_samples);
                }
                else
                    grad.col(1).setZero();

                return grad;
            }

            // EigenFunction& eigenFunction()
            // {
            //     return _f;
            // }

            // template <typename... Args>
            // void addEigenPair(const double& eigenvalue, const EigenFunction& eigenfunction, Args... args)
            // {
            //     _d.push_back(eigenvalue);
            //     _f.push_back(eigenfunction);

            //     if constexpr (sizeof...(args) > 0)
            //         addEigenPair(args...);
            // }

            template <typename... Args>
            void addPair(const double& value, const Function& function, Args... args)
            {
                _pairs.insert(std::make_pair(value, function));

                if constexpr (sizeof...(args) > 0)
                    addPair(args...);
            }

        protected:
            double _l;

            // Lift signal and noise variance
            using AbstractKernel<Params, RiemannSqExp<Params, Function>>::_sf2;
            using AbstractKernel<Params, RiemannSqExp<Params, Function>>::_sn2;

            // std::vector<double> _d;
            // std::vector<EigenFunction> _f;

            // Eigen pairs (eigenvalues, eigenfunctions)
            // EigenFunction _f;
            // This map can become ordered
            std::unordered_map<double, Function> _pairs;

            Eigen::VectorXd parameters() const override
            {
                return tools::makeVector(std::log(_l));
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params) override
            {
                _l = std::exp(params(0));
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const override { return 1; }
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_RIEMANNSQEXP_HPP