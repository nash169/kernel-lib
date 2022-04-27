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
                for (size_t k = 0; k < _d.rows(); k++)
                    r += _s(k) * _f[k].template operator()<SIZE>(x) * _f[k].template operator()<SIZE>(y);

                return r;
            }

            template <typename Derived>
            EIGEN_ALWAYS_INLINE auto gradient(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, const size_t& i = 1) const
            {
                Eigen::VectorXd g = Eigen::VectorXd::Zero(x.rows());

                for (size_t k = 0; k < _d.rows(); k++)
                    g += _s(k) * ((i) ? _f[k].template grad<SIZE>(x) * _f[k].template operator()<SIZE>(y) : _f[k].template grad<SIZE>(y) * _f[k].template operator()<SIZE>(x));

                return g;
            }

            template <typename Derived>
            EIGEN_ALWAYS_INLINE auto hessian(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, const size_t& i = 0) const
            {
                Eigen::MatrixXd h = Eigen::MatrixXd::Zero(x.rows(), x.rows());

                for (size_t k = 0; k < _d.rows(); k++) {
                    if (i == 0)
                        h += _s(k) * _f[k].template hess<SIZE>(x) * _f[k].template operator()<SIZE>(y);
                    else if (i == 1)
                        h += _s(k) * _f[k].template grad<SIZE>(x) * _f[k].template grad<SIZE>(y).transpose();
                    else if (i == 2)
                        h += _s(k) * _f[k].template grad<SIZE>(y) * _f[k].template grad<SIZE>(x).transpose();
                    else if (i == 3)
                        h += _s(k) * _f[k].template operator()<SIZE>(x) * _f[k].template hess<SIZE>(y);
                }

                return h;
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
            EIGEN_DEVICE_FUNC Eigen::MatrixXd gram(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x, const Eigen::Matrix<double, Eigen::Dynamic, Size>& y) const
            {
                Eigen::MatrixXd k = Eigen::MatrixXd::Zero(x.rows(), y.rows());

                for (size_t i = 0; i < _d.rows(); i++)
                    k += _sf2 * _s(i) * _f[i].multiEval(x) * _f[i].multiEval(y).transpose();

                if (x.data() == y.data())
                    k.diagonal().array() += _sn2;

                return k;
            }

            template <int Size>
            EIGEN_DEVICE_FUNC Eigen::MatrixXd gramGradParams(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x, const Eigen::Matrix<double, Eigen::Dynamic, Size>& y) const
            {
                size_t x_num = x.rows(), y_num = y.rows();
                Eigen::MatrixXd g = Eigen::MatrixXd::Zero(x.rows() * y.rows(), 3),
                                k(x.rows(), y.rows());
                Eigen::VectorXd ds = spectralGrad();

                // #pragma omp parallel
                for (size_t i = 0; i < _d.rows(); i++) {
                    k = _sf2 * _f[i].multiEval(x) * _f[i].multiEval(y).transpose();
                    g.col(0) += 2 * _s(i) * Eigen::Map<Eigen::VectorXd>(k.data(), x_num * y_num);
                    g.col(2) += ds(i) * Eigen::Map<Eigen::VectorXd>(k.data(), x_num * y_num);
                }

                if (x.data() == y.data()) {
                    k = 2 * _sn2 * Eigen::MatrixXd::Identity(x_num, y_num);
                    g.col(1) = Eigen::Map<Eigen::VectorXd>(k.data(), x_num * y_num);
                }
                else
                    g.col(1).setZero();

                return g;
            }

            template <typename... Args>
            void addPair(const double& value, const Function& function, Args... args)
            {
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
            RiemannSqExp(const double& l) : _l(l) {}

            // Lift signal and noise variance
            using AbstractKernel<Params, RiemannSqExp<Params, Function>>::_sf2;
            using AbstractKernel<Params, RiemannSqExp<Params, Function>>::_sn2;

            // Length scale
            double _l;

            // Eigenvalues and Spectral density
            Eigen::VectorXd _d, _s;

            // Eigenfunctions
            std::vector<Function> _f;

            // Spectral density
            virtual void spectral()
            {
                _s = (-0.5 * std::pow(_l, 2) * _d).array().exp();
                _s /= _s.sum();
            }

            // Gradient spectral density
            virtual Eigen::VectorXd spectralGrad() const
            {
                return std::pow(_l, 2) * _s.array() * ((_d.array() * _s.array()).sum() - _d.array());
            }

            Eigen::VectorXd parameters() const override
            {
                return tools::makeVector(std::log(_l));
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params) override
            {
                // Set kernel length
                _l = std::exp(params(0));
                // Update kernel length
                spectral();
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const override { return 1; }
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_RIEMANNSQEXP_HPP