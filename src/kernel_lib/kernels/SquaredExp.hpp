#ifndef KERNEL_LIB_SQUARED_EXP
#define KERNEL_LIB_SQUARED_EXP

#include "kernel_lib/kernels/AbstractKernel2.hpp"
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
        class SquaredExp : public AbstractKernel2<Params> {
        public:
            SquaredExp() : _l(std::exp(Params::exp_sq::l()))
            {
                AbstractKernel2<Params>::_params = Eigen::VectorXd(this->sizeParams());

                AbstractKernel2<Params>::init();

                AbstractKernel2<Params>::_params(2) = Params::exp_sq::l();
            }

            /* Gradient */
            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i = 0) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd grad(x_samples * y_samples, n_features);

                double d = -0.5 / std::pow(_l, 2), s2i = ((i) ? 1 : -1) / std::pow(_l, 2) * AbstractKernel2<Params>::_sf2;

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        grad.row(j * x_samples + i) = s2i * (x.row(i) - y.row(j)) * std::exp((x.row(i) - y.row(j)).squaredNorm() * d);

                return grad;
            }

            /* Hessian */
            Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i = 0) const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

            /* Get specific kernel parameters gradient */
            Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd grad(x_samples * y_samples, 3);

                double l2_i = 1 / std::pow(_l, 2), d = -0.5 * l2_i, m = AbstractKernel2<Params>::_sf2 * l2_i,
                       sf_d = 2 * AbstractKernel2<Params>::_sf2, sn_d = 2 * AbstractKernel2<Params>::_sn2,
                       n2, k;

                size_t index;

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++) {
                        n2 = (x.row(i) - y.row(j)).squaredNorm();
                        k = std::exp(n2 * d);

                        index = j * x_samples + i;
                        grad(index, 0) = sf_d * k;
                        grad(index, 1) = (j == i && &x == &y) ? sn_d : 0;
                        grad(index, 2) = m * n2 * k;
                    }

                return grad;
            }

            Eigen::MatrixXd hessianParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

        protected:
            double _l;

            // inline void foo (const char) __attribute__((always_inline)); Strong Inline

            /* Kernel */
            Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd k(x_samples, y_samples);

                double d = -0.5 / std::pow(_l, 2);

#ifndef PARALLEL
                k = -2 * x * y.transpose();
                k.colwise() += x.array().pow(2).rowwise().sum().matrix();
                k.rowwise() += y.array().pow(2).rowwise().sum().matrix().transpose();

                k *= d;

                k = AbstractKernel2<Params>::_sf2 * k.array().exp();

                if (&x == &y)
                    k.diagonal().array() += AbstractKernel2<Params>::_sn2;
#else
#ifdef EIGEN_USE_MKL_ALL
#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y.rows(); j++)
                    for (size_t i = 0; i < x.rows(); i++)
                        k(i, j) = (x.row(i) - y.row(j)).squaredNorm() * d;

                k = AbstractKernel2<Params>::_sf2 * k.array().exp();

                if (&x == &y)
                    k.diagonal().array() += AbstractKernel2<Params>::_sn2;
#else
#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        k(i, j) = AbstractKernel2<Params>::_sf2 * std::exp((x.row(i) - y.row(j)).squaredNorm() * d)
                            + ((j == i && &x == &y) ? AbstractKernel2<Params>::_sn2 + 1e-8 : 0);
#endif
#endif
                return k;
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params)
            {
                AbstractKernel2<Params>::_params(2) = params(0);
                _l = std::exp(params(0));
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const { return 1; }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNEL_LIB_SQUARED_EXP