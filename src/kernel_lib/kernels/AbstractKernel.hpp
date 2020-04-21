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

            /* Data */
            std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> data()
            {
                return std::make_tuple(_x, _y);
            }

            void setData(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                _n_features = x.cols();
                REQUIRED_DIMENSION(_n_features == y.cols(), "Y must have the same dimension of X")

                _x_samples = x.rows();
                _y_samples = y.rows();

                _x = x;
                _y = y;
            }

            /* Parameters */
            Eigen::VectorXd params()
            {
                return static_cast<const Kernel*>(this)->parameters();
            }

            void setParams(const Eigen::VectorXd& params)
            {
                static_cast<Kernel*>(this)->setParameters(params);
            }

            Eigen::MatrixXd gradParams()
            {
                return static_cast<const Kernel*>(this)->gradientParams();
            }

            size_t sizeParams()
            {
                return static_cast<const Kernel*>(this)->sizeParameters();
            }

            /* Settings */
            // Settings are specific for each kernel

            /* Evaluate Kernel */
            Eigen::VectorXd
            operator()() const
            {
                return static_cast<const Kernel*>(this)->kernel();
            }

            Eigen::VectorXd operator()(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                setData(x, y);

                return (*this)();
            }

            /* Evaluate Gradient */
            Eigen::MatrixXd grad() const
            {
                return static_cast<const Kernel*>(this)->gradient();
            }

            Eigen::MatrixXd grad(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                setData(x, y);

                return this->grad();
            }

            /* Evaluate Hessian */
            Eigen::MatrixXd hess() const
            {
                return static_cast<const Kernel*>(this)->hessian();
            }

            Eigen::MatrixXd hess(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                setData(x, y);

                return this->hess();
            }

        protected:
            size_t _n_features, _x_samples, _y_samples;

            double _sigma_f, _sigma_n;

            Eigen::MatrixXd _x, _y;
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP