#ifndef KERNELLIB_KERNELS_RIEMANNMATERN_HPP
#define KERNELLIB_KERNELS_RIEMANNMATERN_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

#include "kernel_lib/tools/helper.hpp"

namespace kernel_lib {

    namespace defaults {
        struct riemann_matern {
            // Log length
            PARAM_SCALAR(double, l, 1);

            // Topological dimension
            PARAM_SCALAR(double, d, 3);

            //
            PARAM_SCALAR(double, nu, 1.5);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params, typename EigenFunction>
        class RiemannMatern : public AbstractKernel<Params, RiemannMatern<Params, EigenFunction>> {
        public:
            RiemannMatern() : _l(std::exp(Params::riemann_matern::l())), _d(Params::riemann_matern::d()), _nu(Params::riemann_matern::nu())
            {
                /* Normalization parameters*/
                _n = 0;

                for (auto& i : _f.eigenPair())
                    _n += std::pow(1 / std::pow(_l, 2) + i.first, -_nu - _d * 0.5);

                _n /= _f.eigenPair().size();
            }

            /* Overload kernel for handling single sample evaluation */
            template <typename Derived>
            inline __attribute__((always_inline)) double kernel(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                double r = 0;

                for (auto& i : _f.eigenPair())
                    r += std::pow(1 / std::pow(_l, 2) + i.first, -_nu - _d * 0.5) * _f(x, i.first) * _f(y, i.first);

                return r / _n;
            }

        protected:
            double _l, _nu, _d, _n;

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
                    _n += std::pow(1 / std::pow(_l, 2) + i.first, -_nu - _d * 0.5);

                _n /= _f.eigenPair().size();
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const { return 1; }
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_RIEMANNMATERN_HPP