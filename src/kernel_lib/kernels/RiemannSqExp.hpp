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

            template <typename Derived>
            EIGEN_ALWAYS_INLINE double kernel(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                double r = 0;

                // this has to parallelized
                for (size_t i = 0; i < _d.rows(); i++)
                    r += _s(i)
                                * _f[i].template operator()
                            < (Derived::RowsAtCompileTime == 1)
                        ? Derived::ColsAtCompileTime
                        : Derived::RowsAtCompileTime > (x)*_f[i].template operator() < (Derived::RowsAtCompileTime == 1) ? Derived::ColsAtCompileTime
                                                                                                                         : Derived::RowsAtCompileTime > (y);

                return r;
            }

            template <int Size>
            EIGEN_DEVICE_FUNC Eigen::MatrixXd gram(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x, const Eigen::Matrix<double, Eigen::Dynamic, Size>& y) const
            {
                Eigen::MatrixXd k = Eigen::MatrixXd::Zero(x.rows(), y.rows());

                for (size_t i = 0; i < _d.rows(); i++)
                    k += _s(i) * _f[i].multiEval(x) * _f[i].multiEval(y).transpose();

                if (x.data() == y.data())
                    k.diagonal().array() += _sn2;

                return k;
            }

            template <typename Derived>
            EIGEN_ALWAYS_INLINE auto gradient(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, const size_t& i = 1) const
            {
                Eigen::VectorXd g = Eigen::VectorXd::Zero(x.rows());

                for (size_t i = 0; i < _d.rows(); i++)
                    g += _s(i) * ((i) ? _f[i].template grad<SIZE>(x) * _f[i].template operator()<SIZE>(y) : _f[i].template grad<SIZE>(y) * _f[i].template operator()<SIZE>(x));

                return g;
            }

            template <typename Derived>
            EIGEN_ALWAYS_INLINE double gradientParams(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, const size_t& i = 1) const
            {
                double g = 0;
                Eigen::VectorXd ds = spectralGrad();

                for (size_t i = 0; i < _d.rows(); i++)
                    g += ds(i) * _f[i].template operator()<SIZE>(x) * _f[i].template operator()<SIZE>(y);

                return g;
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

            template <typename... Args>
            void addPair(const double& value, const Function& function, Args... args)
            {
                _pairs.insert(std::make_pair(value, function));

                // Insert eigenvalue
                _d.conservativeResize(_d.rows() + 1);
                _d(_d.rows() - 1) = value;

                // Insert eigenfunction
                _f.push_back(function);

                if constexpr (sizeof...(args) > 0)
                    addPair(args...);

                // Update spectral density
                spectral();
            }

        protected:
            double _l;

            // Lift signal and noise variance
            using AbstractKernel<Params, RiemannSqExp<Params, Function>>::_sf2;
            using AbstractKernel<Params, RiemannSqExp<Params, Function>>::_sn2;

            // Eigenvalues and Spectral density
            Eigen::VectorXd _d, _s;

            // Eigenfunctions
            std::vector<Function> _f;

            // Eigen pairs (eigenvalues, eigenfunctions)
            // EigenFunction _f;
            // This map can become ordered
            std::unordered_map<double, Function> _pairs;

            // Spectral density
            virtual void spectral()
            {
                _s = (-0.5 * std::pow(_l, 2) * _d).array().exp();
                _s *= _sf2 / _s.sum();
            }

            // Gradient spectral density
            virtual Eigen::VectorXd spectralGrad() const
            {
                Eigen::VectorXd ds = _l * _s;
                return _s.array() * (ds.sum() - ds.array());
            }

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