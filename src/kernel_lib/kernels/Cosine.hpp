#ifndef KERNELLIB_KERNELS_COSINE_HPP
#define KERNELLIB_KERNELS_COSINE_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel_cos {
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class Cosine : public AbstractKernel<Params, Cosine<Params>> {
        public:
            Cosine() {}

            /* Evaluate Kernel */
            Eigen::VectorXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                size_t index = 0, x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();
                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::VectorXd ker(x_samples * y_samples);

                for (size_t i = 0; i < x_samples; i++) {
                    for (size_t j = 0; j < y_samples; j++) {
                        ker.row(index) = x.row(i) * y.row(j).transpose() / (x.row(i).norm() * y.row(j).norm());
                        if (std::isnan(ker(index)))
                            ker(index) = 1;
                        index++;
                    }
                }

                return ker;
            }

            /* Evaluate Gradient */
            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd grad;

                return grad;
            }

            /* Evaluate Hessian */
            Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

            /* Parameters */
            Eigen::VectorXd parameters() const
            {
                Eigen::VectorXd params;

                return params;
            }

            void setParameters(const Eigen::VectorXd& params) {}

            Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd grad_params;

                return grad_params;
            }

            /* Settings */

        protected:
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_COSINE_HPP