#ifndef KERNEL_LIB_SQUARED_EXP
#define KERNEL_LIB_SQUARED_EXP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/helper.hpp"

namespace kernel_lib {
    namespace defaults {
        struct exp_sq {
            // Log length
            PARAM_SCALAR(double, l, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class SquaredExp : public AbstractKernel<Params, SquaredExp<Params>> {
        public:
            SquaredExp() : _l(std::exp(Params::exp_sq::l()))
            {
            }

            /* Overload kernel for handling single sample evaluation */
            template <typename Derived>
            inline __attribute__((always_inline)) double kernel(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                return std::exp((x - y).squaredNorm() * -0.5 / std::pow(_l, 2));
            }

            /* Overload gradient for handling single sample evaluation (fastest solution but there is some issue for inferring size) */
            template <typename Derived>
            inline __attribute__((always_inline)) auto gradient(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, const size_t& i = 1) const
            {
                return ((i) ? (x - y) : (y - x)) / std::pow(_l, 2) * kernel(x, y);
            }

            template <typename Derived>
            inline __attribute__((always_inline)) double gradientParams(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, const size_t& i = 1) const
            {
                return (x - y).squaredNorm() / std::pow(_l, 2) * kernel(x, y);
            }

        protected:
            double _l;

            Eigen::VectorXd parameters() const override
            {
                return tools::makeVector(std::log(_l));
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params) override
            {
                _l = std::exp(params(0));
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const override { return 1; }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNEL_LIB_SQUARED_EXP