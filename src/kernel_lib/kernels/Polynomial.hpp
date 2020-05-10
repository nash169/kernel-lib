#ifndef KERNELLIB_KERNELS_POLYNOMIAL_HPP
#define KERNELLIB_KERNELS_POLYNOMIAL_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel_poly {
            PARAM_SCALAR(double, constant, 0);
            PARAM_SCALAR(int, degree, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class Polynomial : public AbstractKernel {
        public:
            Polynomial() : _const(Params::kernel_poly::constant()), _degree(Params::kernel_poly::degree()) {}

            Eigen::VectorXd dotProduct(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                size_t index = 0, x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();
                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::VectorXd dot_prod(x_samples * y_samples);

                for (size_t i = 0; i < x_samples; i++) {
                    for (size_t j = 0; j < y_samples; j++) {
                        dot_prod(index) = x.row(i) * y.row(j).transpose() + _const;
                        index++;
                    }
                }

                return dot_prod;
            }

        protected:
            double _const;

            int _degree;

            Eigen::VectorXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                return dotProduct(x, y).array().pow(_degree);
            }

            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                size_t index = 0, x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();
                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::VectorXd grad(x_samples * y_samples);

                for (size_t i = 0; i < x_samples; i++) {
                    for (size_t j = 0; j < y_samples; j++) {
                        grad.row(index) = _degree * std::pow(x.row(i) * y.row(j).transpose() + _const, _degree - 1) * x.row(i);
                        index++;
                    }
                }

                return _degree * dotProduct(x, y).array().pow(_degree - 1);
            }

            Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                Eigen::MatrixXd hess;

                return hess;
            }

            Eigen::VectorXd parameters() const override
            {
                Eigen::VectorXd params;

                return params;
            }

            void setParameters(const Eigen::VectorXd& params) override {}

            Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                Eigen::MatrixXd grad_params;

                return grad_params;
            }

            size_t sizeParameters() const override
            {
                return 0;
            }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_POLYNOMIAL_HPP