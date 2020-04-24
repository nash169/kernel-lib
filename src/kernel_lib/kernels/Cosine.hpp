#ifndef KERNELLIB_KERNELS_COSINE_HPP
#define KERNELLIB_KERNELS_COSINE_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel_cos {
            /* data */
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class Cosine : public AbstractKernel<Params, Cosine<Params>> {

            using Kernel_t = AbstractKernel<Params, Cosine<Params>>;

        public:
            Cosine() {}

            /* Parameters */
            Eigen::VectorXd parameters() const
            {
                Eigen::VectorXd params;

                return params;
            }

            void setParameters(const Eigen::VectorXd& params)
            {
            }

            Eigen::MatrixXd gradientParams() const
            {
                Eigen::MatrixXd grad_params;

                return grad_params;
            }

            /* Evaluate Kernel */
            Eigen::VectorXd kernel() const
            {
                Eigen::VectorXd ker(Kernel_t::_x_samples * Kernel_t::_y_samples);
                size_t index = 0;

                for (size_t i = 0; i < Kernel_t::_x_samples; i++) {
                    for (size_t j = 0; j < Kernel_t::_y_samples; j++) {
                        ker(index) = Kernel_t::_x.row(i) * Kernel_t::_y.row(j).transpose() / Kernel_t::_x.row(i).norm() / Kernel_t::_y.row(j).norm();
                        index++;
                    }
                }

                return ker;
            }

            /* Evaluate Gradient */
            Eigen::MatrixXd gradient() const
            {
                Eigen::MatrixXd grad;

                return grad;
            }

            /* Evaluate Hessian */
            Eigen::MatrixXd hessian() const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

            /* Settings */

        protected:
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_COSINE_HPP