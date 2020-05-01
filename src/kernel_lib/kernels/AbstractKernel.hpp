#ifndef KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP
#define KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP

#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {
    namespace kernels {
        class AbstractKernel {
        public:
            AbstractKernel() {}

            /* Evaluate Kernel */
            Eigen::VectorXd operator()(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                return this->kernel(x, y);
            }

            virtual Eigen::VectorXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;

            /* Evaluate Gradient */
            Eigen::MatrixXd grad(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                return this->gradient(x, y);
            }

            virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;

            /* Evaluate Hessian */
            Eigen::MatrixXd hess(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                return this->hessian(x, y);
            }

            virtual Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;

            /* Parameters */
            Eigen::VectorXd params()
            {
                return this->parameters();
            }

            virtual Eigen::VectorXd parameters() const = 0;

            /* Set parameters */
            void setParams(const Eigen::VectorXd& params)
            {
                this->setParameters(params);
            }

            virtual void setParameters(const Eigen::VectorXd& params) = 0;

            /* Parameters' gradient */
            Eigen::MatrixXd gradParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                return this->gradientParams(x, y);
            }

            virtual Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;

            /* Parameters' size */
            size_t sizeParams()
            {
                return this->sizeParameters();
            }

            virtual size_t sizeParameters() const = 0;
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP