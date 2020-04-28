#ifndef KERNELLIB_KERNELS_EXPVELOCITYDIRECTED_HPP
#define KERNELLIB_KERNELS_EXPVELOCITYDIRECTED_HPP

#include "kernel_lib/kernels/Exp.hpp"
#include "kernel_lib/kernels/Polynomial.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel_expvel_dir {
            /* data */
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class ExpVelocityDirected : public AbstractKernel<Params, ExpVelocityDirected<Params>> {
        public:
            ExpVelocityDirected() : _exp(), _cosine() {}

            /* Evaluate Kernel */
            Eigen::VectorXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                return _exp(x, y) + _cosine(x, y);
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
            Exp<Params> _exp;
            Polynomial<Params> _cosine;
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_EXPVELOCITYDIRECTED_HPP