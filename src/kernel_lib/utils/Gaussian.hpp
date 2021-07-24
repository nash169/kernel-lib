#ifndef KERNELLIB_UTILS_GAUSSIAN_HPP
#define KERNELLIB_UTILS_GAUSSIAN_HPP

#include <math.h>

#include "kernel_lib/kernels/SquaredExp.hpp"
#include "kernel_lib/utils/Expansion.hpp"

namespace kernel_lib {
    namespace defaults {
        struct gaussian {
            PARAM_VECTOR(double, mu, 0, 0);
        };
    } // namespace defaults

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
            Eigen::VectorXd multiEval(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x) const
            {
                Eigen::VectorXd r(x.rows());

#pragma omp parallel for
                for (size_t i = 0; i < r.rows(); i++)
                    r(i) = this->operator()<Size>(x.row(i));

                return r;
            }

            Gaussian& setParams(const Eigen::VectorXd& params)
            {
                if constexpr (std::is_same_v<Kernel, kernels::SquaredExp<Params>>) {
                    _kernel.setParameters(params(0));
                    _mu = params.segment(1, params.size() - 1);
                    _weight = 1 / (params(0) * std::sqrt(2 * M_PI));
                }
                else if (std::is_same_v<Kernel, kernels::SquaredExpFull<Params>>) {
                    _kernel.setParameters(params.segment(0, _kernel.sizeParameters() - 1));
                    _mu = params.segment(_kernel.sizeParameters(), params.size() - 1);

                    double det = Eigen::Map<Eigen::MatrixXd>(params.segment(0, _kernel.sizeParameters() - 1).data(), _mu.size(), _mu.size()).determinant();
                    _weight = 1 / std::sqrt(std::pow(2 * M_PI, _mu.size()) * det);
                }
                else {
                    _weight = 1;
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