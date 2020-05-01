#ifndef KERNELLIB_KERNELS_EXPVELOCITYDIRECTED_HPP
#define KERNELLIB_KERNELS_EXPVELOCITYDIRECTED_HPP

#include "kernel_lib/kernels/Cosine.hpp"
#include "kernel_lib/kernels/Exp.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel_exp_vel {
            // Settings
            PARAM_SCALAR(double, angle_ref, M_PI);
        };
    } // namespace defaults
    namespace kernels {
        template <typename Params>
        class ExpVelocityDirected : public AbstractKernel {
        public:
            ExpVelocityDirected() : _exp(), _cosine(), _angle_ref(Params::kernel_exp_vel::angle_ref())
            {
                _upper_limit = 3 * Params::kernel_exp::sigma()(0);
            }

            /* Evaluate Kernel */
            Eigen::VectorXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                size_t m_x = x.rows(), m_y = y.rows(), d = x.cols() / 2;
                return _exp(x.block(0, 0, m_x, d), y.block(0, 0, m_y, d)).array() * tools::linearMap(_cosine(x.block(0, d, m_x, d), y.block(0, d, m_y, d)), std::cos(_angle_ref), 1, _upper_limit, 0).array().exp();
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
                return _exp.sizeParameters() + _cosine.sizeParameters();
            }

        protected:
            double _angle_ref, _upper_limit;

            Exp<Params> _exp;
            Cosine<Params> _cosine;
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_EXPVELOCITYDIRECTED_HPP