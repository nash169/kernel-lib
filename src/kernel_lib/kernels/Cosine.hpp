#ifndef KERNELLIB_KERNELS_COSINE_HPP
#define KERNELLIB_KERNELS_COSINE_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {

    namespace defaults {
        struct cosine {
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class Cosine : public AbstractKernel<Params> {
        public:
            Cosine() : _nan_default(1)
            {
                AbstractKernel<Params>::init();
            }

            /* Evaluate gradient */
            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd grad(x_samples * y_samples, n_features);

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
            double _nan_default;

            /* Kernel */
            Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd k(x_samples, y_samples);

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++) {
                        k(i, j) = AbstractKernel<Params>::_sf2 * x.row(i) * y.row(j).transpose() / (x.row(i).norm() * y.row(j).norm())
                            + ((j == i && &x == &y) ? AbstractKernel<Params>::_sn2 + 1e-8 : 0);
                        if (std::isnan(k(i, j)))
                            k(i, j) = _nan_default;
                    }

                return k;
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params)
            {
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const { return 0; }
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_COSINE_HPP