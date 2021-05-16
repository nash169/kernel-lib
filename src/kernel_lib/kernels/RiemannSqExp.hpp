#ifndef KERNELLIB_KERNELS_RIEMANNSQEXP_HPP
#define KERNELLIB_KERNELS_RIEMANNSQEXP_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

#include "kernel_lib/tools/helper.hpp"

namespace kernel_lib {

    namespace defaults {
        struct riemann_exp_sq {
            // Log length
            PARAM_SCALAR(double, l, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params, typename EigenFunction>
        class RiemannSqExp : public AbstractKernel<Params, RiemannSqExp<Params, EigenFunction>> {
        public:
            RiemannSqExp() : _l(std::exp(Params::riemann_exp_sq::l()))
            {
                /* Normalization parameters*/
                _n = 0;

                for (auto& i : _f.eigenPair())
                    _n += std::exp(-0.5 / std::pow(_l, 2) * i.first);

                _n /= _f.eigenPair().size();
            }

            /* Overload kernel for handling single sample evaluation */
            template <typename Derived>
            inline __attribute__((always_inline)) double kernel(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                double r = 0;

                for (auto& i : _f.eigenPair())
                    r += std::exp(-0.5 / std::pow(_l, 2) * i.first) * _f(x, i.first) * _f(y, i.first);

                return r / _n;
            }

            template <typename Derived>
            inline __attribute__((always_inline)) double gradientParams(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, const size_t& i = 1) const
            {
                double r = 0;

                for (auto& i : _f.eigenPair())
                    r -= std::pow(_l, 2) * i.first * std::exp(-0.5 / std::pow(_l, 2) * i.first) * _f(x, i.first) * _f(y, i.first);

                return r / _n;
            }

        protected:
            double _l, _n;

            // Eigen pairs (eigenvalues, eigenfunctions)
            EigenFunction _f;

            Eigen::VectorXd parameters() const override
            {
                return tools::makeVector(std::log(_l));
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params)
            {
                _l = std::exp(params(0));

                /* Normalization parameters*/
                _n = 0;

                for (auto& i : _f.eigenPair())
                    _n += std::exp(-0.5 / std::pow(_l, 2) * i.first);

                _n /= _f.eigenPair().size();
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const { return 1; }
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_RIEMANNSQEXP_HPP