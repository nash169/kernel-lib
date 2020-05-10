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
            }

            ExpVelocityDirected& setAngle(double angle_ref)
            {
                _angle_ref = angle_ref;

                return *this;
            }

        protected:
            double _angle_ref;

            Exp<Params> _exp;
            Cosine<Params> _cosine;

            Eigen::VectorXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                size_t m_x = x.rows(), m_y = y.rows(), d = x.cols() / 2;
                double sigma = _exp.params()(0);

                return (_exp.log_kernel(x.block(0, 0, m_x, d), y.block(0, 0, m_y, d)).array() - tools::linearMap(_cosine(x.block(0, d, m_x, d), y.block(0, d, m_y, d)), std::cos(_angle_ref), 1, 3 * sigma, 0).array().pow(2) * 0.5 / std::pow(sigma, 2)).exp();
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
                return _exp.params();
            }

            void setParameters(const Eigen::VectorXd& params) override
            {
                _exp.setParams(params);
            }

            Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                Eigen::MatrixXd grad_params;

                return grad_params;
            }

            size_t sizeParameters() const override
            {
                return _exp.sizeParams() + _cosine.sizeParams();
            }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_EXPVELOCITYDIRECTED_HPP