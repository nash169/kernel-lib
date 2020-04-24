#ifndef KERNELLIB_KERNELS_EXPVELOCITYDIRECTED_HPP
#define KERNELLIB_KERNELS_EXPVELOCITYDIRECTED_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel_expvel_dir {
            /* data */
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class ExpVelocityDirected : public AbstractKernel<Params, ExpVelocityDirected<Params>> {

            using Kernel_t = AbstractKernel<Params, ExpVelocityDirected<Params>>;

        public:
            ExpVelocityDirected() {}

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
                Eigen::VectorXd ker;

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

#endif // KERNELLIB_KERNELS_EXPVELOCITYDIRECTED_HPP