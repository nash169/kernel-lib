#ifndef KERNELLIB_KERNELS_EXPGRADIENTDIRECTED_HPP
#define KERNELLIB_KERNELS_EXPGRADIENTDIRECTED_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel_expgrad_dir {
            /* data */
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class ExpGradientDirected : public AbstractKernel<Params, ExpGradientDirected<Params>> {
        public:
            ExpGradientDirected() {}

            /* Evaluate Kernel */
            Eigen::VectorXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::VectorXd ker;

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

            void setParameters(const Eigen::VectorXd& params)
            {
            }

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

#endif // KERNELLIB_KERNELS_EXPGRADIENTDIRECTED_HPP