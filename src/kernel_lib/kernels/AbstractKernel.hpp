#ifndef KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP
#define KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP

#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel {
            PARAM_SCALAR(double, sigma_f, 1.0);
            PARAM_SCALAR(double, sigma_n, 0.0);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params, typename Kernel>
        class AbstractKernel {
        public:
            AbstractKernel() : _sigma_f(Params::kernel::sigma_f())
            {
                _sigma_n = (Params::kernel::sigma_n() == 0) ? 1e-8 : Params::kernel::sigma_n();
            }

            /* Evaluate Kernel */
            Eigen::VectorXd operator()(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                return static_cast<const Kernel*>(this)->kernel(x, y);
            }

            /* Evaluate Gradient */
            Eigen::MatrixXd grad(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                return this->grad(x, y);
            }

            /* Evaluate Hessian */
            Eigen::MatrixXd hess(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                return static_cast<const Kernel*>(this)->hessian(x, y);
            }

            /* Parameters */
            Eigen::VectorXd params()
            {
                Eigen::VectorXd params(sizeParams());
                params(0) = _sigma_f;
                params(1) = _sigma_n;
                params.segment(2, params.rows() - 2) = static_cast<const Kernel*>(this)->parameters();

                return params;
            }

            void setParams(const Eigen::VectorXd& params)
            {
                _sigma_f = params(0);
                _sigma_n = params(1);
                static_cast<Kernel*>(this)->setParameters(params.segment(2, params.rows() - 2));
            }

            Eigen::MatrixXd gradParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                Eigen::MatrixXd grad_params(x.rows() * y.rows(), sizeParams());

                grad_params.block(0, 0, grad_params.rows(), 1) = 2 * _sigma_f * static_cast<const Kernel*>(this)->kernel(x, y);

                grad_params.block(0, 2, grad_params.rows(), grad_params.cols() - 2) = static_cast<const Kernel*>(this)->gradientParams(x, y);

                return grad_params;
            }

            size_t sizeParams()
            {
                return 2 + static_cast<const Kernel*>(this)->sizeParameters();
            }

        protected:
            double _sigma_f, _sigma_n;
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP