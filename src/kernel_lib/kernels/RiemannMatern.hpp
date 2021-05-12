#ifndef KERNELLIB_KERNELS_RIEMANNMATERN_HPP
#define KERNELLIB_KERNELS_RIEMANNMATERN_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

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
        class RiemannMatern : public AbstractKernel<Params> {
        public:
            RiemannMatern() : _l(std::exp(Params::riemann_matern::l())), _d(Params::riemann_matern::d()), _nu(Params::riemann_matern::nu())
            {
                /* Init parameters vector dimension */
                AbstractKernel<Params>::_params = Eigen::VectorXd(this->sizeParams());

                /* Set signal and noise variance */
                AbstractKernel<Params>::init();

                /* Set specific kernel parameters */
                AbstractKernel<Params>::_params(2) = Params::riemann_matern::l();

                /* Init parameters vector dimension */
                double k = 1 / std::pow(_l, 2), p = -_nu - _d * 0.5;

                /* Normalization parameters*/
                _n = 0;

                for (auto& i : _f.eigenPair())
                    _n += std::pow(k + i.first, p);
            }

            /* Evaluate gradient */
            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const
            {
                Eigen::MatrixXd grad;

                return grad;
            }

            /* Evaluate hessian */
            Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

            /* Evaluate parameters gradient */
            Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd grad;

                return grad;
            }

            /* Evaluate parameters hessian */
            Eigen::MatrixXd hessianParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

        protected:
            double _l, _nu, _d, _n;

            /* Kernel */
            Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd k(x_samples, y_samples);

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        k(i, j) = kernel(x.row(i), y.row(i));

                return k;
            }

            /* Overload kernel for handling single sample evaluation */
            template <typename Derived>
            inline __attribute__((always_inline)) double kernel(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                double r = 0;

                for (auto& i : _f.eigenPair()) {
                    double k = 1 / std::pow(_l, 2), p = -_nu - _d * 0.5;
                    r += AbstractKernel<Params>::_sf2 / _n * std::pow(k + i.first, p) * _f(x, i.first) * _f(y, i.first) + ((&x == &y) ? AbstractKernel<Params>::_sn2 + 1e-8 : 0);
                }

                return r;
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params)
            {
                AbstractKernel<Params>::_params(2) = params(0);
                _l = std::exp(params(0));
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const { return 1; }
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_RIEMANNMATERN_HPP