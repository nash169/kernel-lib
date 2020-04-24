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
        class Polynomial : public AbstractKernel<Params, Polynomial<Params>> {

            using Kernel_t = AbstractKernel<Params, Polynomial<Params>>;

        public:
            Polynomial() {}

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
                return dotProduct().array().pow(_degree);
            }

            /* Evaluate Gradient */
            Eigen::MatrixXd gradient() const
            {
                Eigen::VectorXd grad(Kernel_t::_x_samples * Kernel_t::_y_samples);
                size_t index = 0;

                for (size_t i = 0; i < Kernel_t::_x_samples; i++) {
                    for (size_t j = 0; j < Kernel_t::_y_samples; j++) {
                        grad.row(index) = _degree * std::pow(Kernel_t::_x.row(i) * Kernel_t::_y.row(j).transpose() + _const, _degree - 1) * Kernel_t::_x.row(i);
                        index++;
                    }
                }

                return _degree * dotProduct().array.pow(_degree - 1);
            }

            /* Evaluate Hessian */
            Eigen::MatrixXd hessian() const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

            /* Settings */

        protected:
            double _const;
            int _degree;

            Eigen::VectorXd dotProduct() const
            {
                Eigen::VectorXd dot_prod(Kernel_t::_x_samples * Kernel_t::_y_samples);
                size_t index = 0;

                for (size_t i = 0; i < Kernel_t::_x_samples; i++) {
                    for (size_t j = 0; j < Kernel_t::_y_samples; j++) {
                        dot_prod(index) = Kernel_t::_x.row(i) * Kernel_t::_y.row(j).transpose() + _const;
                        index++;
                    }
                }
            }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_POLYNOMIAL_HPP