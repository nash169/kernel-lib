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

#ifndef KERNEL_LIB_SQUARED_EXP
#define KERNEL_LIB_SQUARED_EXP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/helper.hpp"

// Here it can be avoided to include the header and just init the class
#include "kernel_lib/utils/Gaussian.hpp"

namespace kernel_lib {
    namespace defaults {
        struct exp_sq {
            // Log length
            PARAM_SCALAR(double, l, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class SquaredExp : public AbstractKernel<Params, SquaredExp<Params>> {
        public:
            SquaredExp() : _l(std::exp(Params::exp_sq::l()))
            {
            }

            /* Overload kernel for handling single sample evaluation */
            template <typename Derived>
            EIGEN_ALWAYS_INLINE double kernel(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                return std::exp((x - y).squaredNorm() * -0.5 / std::pow(_l, 2));
            }

            /* Overload gradient for handling single sample evaluation (fastest solution but there is some issue for inferring size) */
            template <typename Derived>
            EIGEN_ALWAYS_INLINE auto gradient(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, const size_t& i = 1) const
            {
                return ((i) ? (y - x) : (x - y)) / std::pow(_l, 2) * kernel(x, y);
            }

            template <typename Derived>
            EIGEN_ALWAYS_INLINE auto hessian(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, const size_t& i = 3) const
            {
                Eigen::MatrixXd h(x.size(), x.size());

                if (i == 0 || i == 3) {
                    h = (x - y) * (x - y).transpose() / std::pow(_l, 4);
                    h.diagonal().array() -= 1 / std::pow(_l, 2);
                }
                else {
                    h = (y - x) * (x - y).transpose() / std::pow(_l, 4);
                    h.diagonal().array() += 1 / std::pow(_l, 2);
                }

                h *= kernel(x, y);

                return h;
            }

            template <typename Derived>
            EIGEN_ALWAYS_INLINE double gradientParams(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                return (x - y).squaredNorm() / std::pow(_l, 2) * kernel(x, y);
            }

            friend class utils::Gaussian<Params, SquaredExp<Params>>;

        protected:
            double _l;

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

            /* Kernel logarithms (mainly used by the Gaussian to produce the log-likelihood) */
            template <int Size>
            EIGEN_ALWAYS_INLINE double logKernel(const Eigen::Matrix<double, Size, 1>& x, const Eigen::Matrix<double, Size, 1>& y) const
            {
                return (x - y).squaredNorm() * -0.5 / std::pow(_l, 2);
            }

            template <int Size>
            EIGEN_ALWAYS_INLINE auto logGradient(const Eigen::Matrix<double, Size, 1>& x, const Eigen::Matrix<double, Size, 1>& y, const size_t& i = 1) const
            {
                return ((i) ? (y - x) : (x - y)) / std::pow(_l, 2);
            }

            template <int Size>
            EIGEN_ALWAYS_INLINE auto logGradientParams(const Eigen::Matrix<double, Size, 1>& x, const Eigen::Matrix<double, Size, 1>& y) const
            {
                return (x - y).squaredNorm() / std::pow(_l, 2);
            }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNEL_LIB_SQUARED_EXP