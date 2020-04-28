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

            /* Evaluate Gradient */
            Eigen::MatrixXd grad(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                return this->gradient(x, y);
            }

            /* Evaluate Hessian */
            Eigen::MatrixXd hess(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                return this->hessian(x, y);
            }

            /* Parameters */
            Eigen::VectorXd params()
            {
                return this->parameters();
            }

            void setParams(const Eigen::VectorXd& params)
            {
                this->setParameters(params);
            }

            Eigen::MatrixXd gradParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                return this->gradientParams(x, y);
            }

            size_t sizeParams()
            {
                return this->sizeParameters();
            }

            virtual Eigen::VectorXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;
            virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;
            virtual Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;
            virtual Eigen::VectorXd parameters() const = 0;
            virtual void setParameters(const Eigen::VectorXd& params) = 0;
            virtual Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;
            virtual size_t sizeParameters() const = 0;
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP