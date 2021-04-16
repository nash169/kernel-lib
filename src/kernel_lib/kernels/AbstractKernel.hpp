#ifndef KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP
#define KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP

#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {

    namespace defaults {
        struct kernel {
            // Parameters
            PARAM_SCALAR(double, sigma_n, 0.0);
            PARAM_SCALAR(double, sigma_f, 1.0);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class AbstractKernel {
        public:
            AbstractKernel() : _sigma_n(Params::kernel::sigma_n()), _sigma_f(Params::kernel::sigma_f()) {}

            /* Evaluate Kernel */
            Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                double sf2 = std::exp(2 * std::log(_sigma_f)), sn2 = std::exp(2 * std::log(_sigma_n));

                Eigen::MatrixXd k = this->kernel(x, y);

                k *= sf2;

                if (&x == &y)
                    k.diagonal().array() += sn2 + 1e-8;

                return std::move(k);
            }

            /* Evaluate Gradient */
            Eigen::MatrixXd grad(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i = 1) const { return this->gradient(x, y, i); }

            /* Evaluate Hessian */
            Eigen::MatrixXd hess(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const { return this->hessian(x, y); }

            /* Parameters */
            Eigen::VectorXd params() const { return this->parameters(); }

            /* Set parameters */
            void setParams(const Eigen::VectorXd& params)
            {
                _sigma_n = params(0);
                _sigma_f = params(1);

                this->setParameters(params.segment(2, params.rows() - 2));
            }

            /* Parameters' gradient */
            Eigen::MatrixXd gradParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const { return this->gradientParams(x, y); }

            /* Parameters' size */
            size_t sizeParams() const { return this->sizeParameters() + 2; }

        protected:
            double _sigma_n, _sigma_f;

            /* Kernel */
            virtual Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;

            /* Gradient */
            virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const = 0;

            /* Hessian */
            virtual Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const = 0;

            /* Get specific kernel parameters */
            virtual Eigen::VectorXd parameters() const = 0;

            /* Set specific kernel parameters */
            virtual void setParameters(const Eigen::VectorXd& params) = 0;

            /* Get specific kernel parameters gradient */
            virtual Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;

            /* Get number of parameters for the specific kernel */
            virtual size_t sizeParameters() const = 0;
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP