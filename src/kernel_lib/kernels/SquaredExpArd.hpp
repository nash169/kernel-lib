#ifndef KERNEL_LIB_KERNELS_SQUARED_EXP_ARD
#define KERNEL_LIB_KERNELS_SQUARED_EXP_ARD

#include "kernel_lib/kernels/AbstractKernel2.hpp"
#include "kernel_lib/tools/helper.hpp"

namespace kernel_lib {
    namespace defaults {
        struct exp_sq_ard {
            // Log length
            PARAM_VECTOR(double, l, 1, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class SquaredExpArd : public AbstractKernel2<Params> {
        public:
            SquaredExpArd() : _l(Params::exp_sq_ard::l().array().exp())
            {
                AbstractKernel2<Params>::_params = Eigen::VectorXd(this->sizeParams());

                AbstractKernel2<Params>::init();

                AbstractKernel2<Params>::_params.segment(2, _l.rows()) = Params::exp_sq_ard::l();
            }

            /* Gradient */
            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i = 0) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd grad(x_samples * y_samples, n_features);

                Eigen::Array<double, 1, Eigen::Dynamic> d = -0.5 * _l.array().pow(2).inverse(),
                                                        s2i = ((i) ? AbstractKernel2<Params>::_sf2 : -AbstractKernel2<Params>::_sf2) * _l.array().pow(2).inverse();

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        grad.row(j * x_samples + i) = s2i * (x.row(i) - y.row(j)).array() * std::exp(((x.row(i) - y.row(j)).array().square() * d).sum());

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

                Eigen::MatrixXd grad(x_samples * y_samples, 2 + _l.rows());

                double sf_d = 2 * AbstractKernel2<Params>::_sf2, sn_d = 2 * AbstractKernel2<Params>::_sn2;
                Eigen::Array<double, 1, Eigen::Dynamic> l2_i = _l.array().pow(2).inverse(), d = -0.5 * l2_i, m = AbstractKernel2<Params>::_sf2 * l2_i;

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++) {
                        Eigen::Array<double, 1, Eigen::Dynamic> n2 = (x.row(i) - y.row(j)).array().square();
                        double k = std::exp((n2 * d).sum());
                        grad.row(j * x_samples + i) << sf_d * k, (j == i && &x == &y) ? sn_d : 0, m * n2 * k;
                    }

                return grad;
            }

            Eigen::MatrixXd hessianParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

        protected:
            double _dim;
            Eigen::VectorXd _l;

            /* Kernel */
            Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd k(x_samples, y_samples);

                Eigen::Array<double, 1, Eigen::Dynamic> d = -0.5 * _l.array().pow(2).inverse();

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        k(i, j) = AbstractKernel2<Params>::_sf2 * std::exp(((x.row(i) - y.row(j)).array().square() * d).sum())
                            + ((j == i && &x == &y) ? AbstractKernel2<Params>::_sn2 + 1e-8 : 0);

                return k;
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params)
            {
                AbstractKernel2<Params>::_params.segment(2, _l.rows()) = params;
                _l = params.array().exp();
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const { return _l.rows(); }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNEL_LIB_KERNELS_SQUARED_EXP_ARD