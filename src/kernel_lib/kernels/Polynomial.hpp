#ifndef KERNELLIB_KERNELS_POLYNOMIAL_HPP
#define KERNELLIB_KERNELS_POLYNOMIAL_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {

    namespace defaults {
        struct polynomial {
            // Log length
            PARAM_SCALAR(double, d, 1);
            PARAM_SCALAR(double, c, 0);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class Polynomial : public AbstractKernel<Params> {
        public:
            Polynomial() : _d(Params::polynomial::d()), _c(Params::polynomial::c())
            {
                AbstractKernel<Params>::init();
            }

            /* Evaluate gradient */
            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd grad(x_samples * y_samples, n_features);

                if (i == 0)
#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y_samples; j++)
                        for (size_t i = 0; i < x_samples; i++)
                            grad.row(j * x_samples + i) = AbstractKernel<Params>::_sf2 * _d * std::pow(x.row(i) * y.row(j).transpose() + _c, _d - 1) * y.row(i);
                else
#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y_samples; j++)
                        for (size_t i = 0; i < x_samples; i++)
                            grad.row(j * x_samples + i) = AbstractKernel<Params>::_sf2 * _d * std::pow(x.row(i) * y.row(j).transpose() + _c, _d - 1) * x.row(i);

                return grad;
            }

            /* Evaluate hessian */
            Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

            /* Evaluate parameters gradient */
            Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd grad;

                return grad;
            }

            /* Evaluate parameters hessian */
            Eigen::MatrixXd hessianParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

        protected:
            double _d, _c;

            /* Kernel */
            Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd k(x_samples, y_samples);

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        k(i, j) = AbstractKernel<Params>::_sf2 * std::pow(x.row(i) * y.row(j).transpose() + _c, _d)
                            + ((j == i && &x == &y) ? AbstractKernel<Params>::_sn2 + 1e-8 : 0);

                return k;
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params)
            {
                AbstractKernel<Params> _params(2) = params(0);
                _d = params(0);

                AbstractKernel<Params> _params(3) = params(1);
                _c = params(1);
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const { return 2; }
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_POLYNOMIAL_HPP