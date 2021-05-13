#ifndef KERNEL_LIB_SQUARED_EXP
#define KERNEL_LIB_SQUARED_EXP

#include "kernel_lib/kernels/BaseKernel.hpp"
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
        class SquaredExp2 : public BaseKernel<Params, SquaredExp2<Params>> {
        public:
            SquaredExp2() : _l(std::exp(Params::exp_sq::l()))
            {
                /* Init parameters vector dimension */
                BaseKernel<Params, SquaredExp2<Params>>::_params = Eigen::VectorXd(this->sizeParams());

                /* Set signal and noise variance */
                BaseKernel<Params, SquaredExp2<Params>>::init();

                /* Set specific kernel parameters */
                BaseKernel<Params, SquaredExp2<Params>>::_params(2) = Params::exp_sq::l();

                /* Init parameters vector dimension */
                _d = -0.5 / std::pow(_l, 2);
                _g = -1 / std::pow(_l, 2) * BaseKernel<Params, SquaredExp2<Params>>::_sf2;
            }

            /* Overload kernel for handling single sample evaluation */
            template <typename Derived>
            inline __attribute__((always_inline)) double kernel(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                // std::cout << "MatrixDerived" << std::endl;
                return std::exp((x - y).squaredNorm() * _d) + ((&x == &y) ? BaseKernel<Params, SquaredExp2<Params>>::_sn2 + 1e-8 : 0);
            }

            /* Overload kernel for handling single sample evaluation (fastest solution but there is some issue for inferring size) */
            template <int size>
            inline __attribute__((always_inline)) double kernel(const Eigen::Matrix<double, size, 1>& x, const Eigen::Matrix<double, size, 1>& y) const
            {
                // std::cout << "MatrixStatic" << std::endl;
                return BaseKernel<Params, SquaredExp2<Params>>::_sf2 * std::exp((x - y).squaredNorm() * _d) + ((&x == &y) ? BaseKernel<Params, SquaredExp2<Params>>::_sn2 + 1e-8 : 0);
            }

        protected:
            double _l, _d, _g;

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params)
            {
                BaseKernel<Params, SquaredExp2<Params>>::_params(2) = params(0);
                _l = std::exp(params(0));
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const { return 1; }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNEL_LIB_SQUARED_EXP