#ifndef KERNELLIB_KERNELS_RIEMANNSQEXP_HPP
#define KERNELLIB_KERNELS_RIEMANNSQEXP_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {

    namespace defaults {
        struct riemann_exp_sq {
            // Log length
            PARAM_SCALAR(double, l, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params, typename EigenFunction>
        class RiemannSqExp : public AbstractKernel<Params> {
        public:
            RiemannSqExp() : _l(std::exp(Params::riemann_exp_sq::l()))
            {
                /* Init parameters vector dimension */
                AbstractKernel<Params>::_params = Eigen::VectorXd(this->sizeParams());

                /* Set signal and noise variance */
                AbstractKernel<Params>::init();

                /* Set specific kernel parameters */
                AbstractKernel<Params>::_params(2) = Params::riemann_exp_sq::l();

                /* Init parameters vector dimension */
                _d = -0.5 / std::pow(_l, 2);

                /* Normalization parameters*/
                _n = 0;

                for (auto& i : _f.eigenPair())
                    _n += std::exp(_d * i.first);
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
            double _l, _d, _n;

            // Eigen pairs (eigenvalues, eigenfunctions)
            EigenFunction _f;

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

                for (auto& i : _f.eigenPair())
                    r += AbstractKernel<Params>::_sf2 / _n * std::exp(_d * i.first) * _f(x, i.first) * _f(y, i.first) + ((&x == &y) ? AbstractKernel<Params>::_sn2 + 1e-8 : 0);

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

#endif // KERNELLIB_KERNELS_RIEMANNSQEXP_HPP