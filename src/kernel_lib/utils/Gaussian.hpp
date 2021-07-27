#ifndef KERNELLIB_UTILS_GAUSSIAN_HPP
#define KERNELLIB_UTILS_GAUSSIAN_HPP

#include <math.h>

#include "kernel_lib/tools/helper.hpp"
#include "kernel_lib/tools/macros.hpp"

// #include "kernel_lib/kernels/SquaredExp.hpp"

namespace kernel_lib {
    namespace defaults {
        struct gaussian {
            PARAM_VECTOR(double, mu, 0, 0);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class SquaredExp;

        template <typename Params>
        class SquaredExpFull;
    } // namespace kernels

    namespace utils {
        template <typename Params, typename Kernel = kernels::SquaredExp<Params>>
        class Gaussian {
        public:
            Gaussian() : _mu(Params::gaussian::mu()), _kernel() {}

            template <int Size>
            EIGEN_ALWAYS_INLINE double operator()(const Eigen::Matrix<double, Size, 1>& x) const
            {
                return _weight * _kernel.template kernelImpl<Size>(x, _mu);
            }

            template <int Size>
            EIGEN_ALWAYS_INLINE double log(const Eigen::Matrix<double, Size, 1>& x) const
            {
                if constexpr (std::is_same_v<Kernel, kernels::SquaredExp<Params>>) {
                    return _kernel.logKernel(x, _mu) - 0.5 * x.size() * (_kernel.parameters() + std::log(2 * M_PI));
                }
                else if constexpr (std::is_same_v<Kernel, kernels::SquaredExpFull<Params>>) {
                    return _kernel.logKernel(x, _mu) - 0.5 * (std::log(_kernel._S.determinant()) + x.size() * std::log(2 * M_PI));
                }
                else {
                    return _kernel.logKernel(x, _mu) + std::log(_weight);
                }
            }

            template <int Size>
            EIGEN_ALWAYS_INLINE auto grad(const Eigen::Matrix<double, Size, 1>& x) const
            {
                return _weight * _kernel.gradient(x, _mu, 0);
            }

            template <int Size>
            EIGEN_ALWAYS_INLINE auto gradParams(const Eigen::Matrix<double, Size, 1>& x) const
            {
                if constexpr (std::is_same_v<Kernel, kernels::SquaredExp<Params>>) {
                    Eigen::VectorXd grad(1 + _mu.size());

                    grad << -0.5 / std::pow(_kernel._l, 2) * Eigen::MatrixXd::Identity(_mu.size(), _mu.size()) + _kernel.logGradientParams(x, _mu),
                        _kernel.logGradient(x, _mu, 1);

                    return grad;
                }
                else if constexpr (std::is_same_v<Kernel, kernels::SquaredExpFull<Params>>) {
                    Eigen::VectorXd grad(_mu.size() + _mu.size() * _mu.size());

                    grad << -0.5 * _kernel._llt.solve(Eigen::MatrixXd::Identity(_mu.size(), _mu.size())) + _kernel.logGradientParams(x, _mu),
                        _kernel.logGradient(x, _mu, 1);

                    return grad;
                }
                else {
                }
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

            auto params()
            {
                if constexpr (std::is_same_v<Kernel, kernels::SquaredExp<Params>>) {
                    Eigen::VectorXd params(1 + _mu.size());
                    params << _kernel.parameters(), _mu;

                    return params;
                }
                else if constexpr (std::is_same_v<Kernel, kernels::SquaredExpFull<Params>>) {
                    size_t size = _mu.size();
                    Eigen::VectorXd params(size + size * size);
                    params << _kernel.parameters(), _mu;

                    return params;
                }
                else {
                    return _weight;
                }
            }

            Gaussian& setParams(const Eigen::VectorXd& params)
            {
                if constexpr (std::is_same_v<Kernel, kernels::SquaredExp<Params>>) {
                    // Set sigma
                    _kernel.setParameters(tools::makeVector(params(0)));

                    // Set mean
                    _mu = params.segment(1, params.size() - 1);

                    // Set normalization
                    _weight = 1 / std::sqrt(std::pow(2 * M_PI * _kernel._l, _mu.size()));
                }
                else if constexpr (std::is_same_v<Kernel, kernels::SquaredExpFull<Params>>) {
                    // Calculate size
                    int size = 0.5 * (std::sqrt(1 + 4 * params.size()) - 1);

                    // Set covariance
                    _kernel.setParameters(params.segment(0, size * size));

                    // Set mean
                    _mu = params.segment(size * size, size);

                    // Set normalization (see SquaredExpFull for the matrix _S)
                    _weight = 1 / std::sqrt(std::pow(2 * M_PI, size) * _kernel._S.determinant());
                    // Eigen::MatrixXd det = Eigen::Map<Eigen::MatrixXd>(params.segment(0, _kernel.sizeParameters() - 1).data(), _mu.size(), _mu.size());
                }
                else {
                    _weight = params(0);
                }

                return *this;
            }

        protected:
            double _weight;

            Eigen::VectorXd _mu;

            Kernel _kernel;
        };
    } // namespace utils
} // namespace kernel_lib

#endif // KERNELLIB_UTILS_GAUSSIAN_HPP