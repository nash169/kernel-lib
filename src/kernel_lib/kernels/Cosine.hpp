#ifndef KERNELLIB_KERNELS_COSINE_HPP
#define KERNELLIB_KERNELS_COSINE_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel_cos {
            // Settings
            PARAM_SCALAR(double, nan_value, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class Cosine : public AbstractKernel {
        public:
            Cosine() : _nan_value(Params::kernel_cos::nan_value())
            {
            }

            Cosine& setNan(double nan_value)
            {
                _nan_value = nan_value;

                return *this;
            }

        protected:
            double _nan_value;

            Eigen::VectorXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                size_t index = 0, x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();
                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::VectorXd ker(x_samples * y_samples);

                for (size_t i = 0; i < x_samples; i++) {
                    for (size_t j = 0; j < y_samples; j++) {
                        ker.row(index) = x.row(i) * y.row(j).transpose() / (x.row(i).norm() * y.row(j).norm());
                        if (std::isnan(ker(index)))
                            ker(index) = _nan_value;
                        index++;
                    }
                }

                return ker;
            }

            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                Eigen::MatrixXd grad;

                return grad;
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

#endif // KERNELLIB_KERNELS_COSINE_HPP