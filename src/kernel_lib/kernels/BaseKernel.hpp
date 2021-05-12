#ifndef KERNELLIB_KERNELS_BASEKERNEL_HPP
#define KERNELLIB_KERNELS_BASEKERNEL_HPP

#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {

    namespace defaults {
        struct kernel {
            // Log signal std
            PARAM_SCALAR(double, sf, 1.0);

            // Log noise std
            PARAM_SCALAR(double, sn, 0.0);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params, typename Kernel>
        class BaseKernel {
        public:
            BaseKernel() : _sf2(std::exp(2 * Params::kernel::sf())), _sn2(std::exp(2 * Params::kernel::sn())) {}

            /* Evaluate kernel */
            // Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const { return this->kernel(x, y); }

            /* Parameters */
            Eigen::VectorXd params() const
            {
                return _params;
            }

            /* Set parameters */
            void setParams(const Eigen::VectorXd& params)
            {
                _params(0) = params(0);
                _params(1) = params(1);

                _sf2 = std::exp(2 * params(0));
                _sn2 = std::exp(2 * params(1));

                this->setParameters(params.segment(2, params.rows() - 2));
            }

            /* Parameters' size */
            size_t sizeParams() const { return this->sizeParameters() + 2; }

        protected:
            double _sf2, _sn2;

            Eigen::VectorXd _params;

            // Init
            void init()
            {
                _params(0) = Params::kernel::sf();
                _params(1) = Params::kernel::sn();
            }

            /* Set specific kernel parameters */
            virtual void setParameters(const Eigen::VectorXd& params) = 0;

            /* Get number of parameters for the specific kernel */
            virtual size_t sizeParameters() const = 0;
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_BASEKERNEL_HPP