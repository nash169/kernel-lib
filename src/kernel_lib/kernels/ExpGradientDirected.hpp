#ifndef KERNELLIB_KERNELS_EXPGRADIENTDIRECTED_HPP
#define KERNELLIB_KERNELS_EXPGRADIENTDIRECTED_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"

namespace kernel_lib {
    namespace kernels {
        template <typename Params>
        class ExpGradientDirected : public AbstractKernel {
        public:
            ExpGradientDirected() {}

            /* Evaluate Kernel */
            Eigen::VectorXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                Eigen::VectorXd ker;

                return ker;
            }

            /* Evaluate Gradient */
            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                Eigen::MatrixXd grad;

                return grad;
            }

            /* Evaluate Hessian */
            Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                Eigen::MatrixXd hess;

                return hess;
            }

            /* Parameters */
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

#endif // KERNELLIB_KERNELS_EXPGRADIENTDIRECTED_HPP