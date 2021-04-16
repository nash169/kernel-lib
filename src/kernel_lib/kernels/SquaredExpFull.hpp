#ifndef KERNEL_LIB_KERNELS_SQUARED_EXP_FULL
#define KERNEL_LIB_KERNELS_SQUARED_EXP_FULL

#include "kernel_lib/kernels/AbstractKernel2.hpp"
#include "kernel_lib/tools/helper.hpp"

namespace kernel_lib {
    namespace defaults {
        struct exp_sq_full {
            // Log length
            PARAM_VECTOR(double, l, 1, 0.5, 0.5, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class SquaredExpFull : public AbstractKernel2<Params> {
        public:
            SquaredExpFull() : _S(Params::exp_sq_full::l())
            {
                _S = _S.reshaped(std::sqrt(_S.rows()), std::sqrt(_S.rows()));

                AbstractKernel2<Params>::_params = Eigen::VectorXd(this->sizeParams());

                AbstractKernel2<Params>::init();

                AbstractKernel2<Params>::_params.segment(2, _S.size()) = Params::exp_sq_full::l();
            }

            /* Gradient */
            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i = 0) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd grad(x_samples * y_samples, n_features);

                Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> U = tools::upperCholesky(_S);

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        grad.row(j * x_samples + i) = AbstractKernel2<Params>::_sf2 * U.solve((x.row(i) - y.row(j)).transpose());
                // U.transpose().solve();
                // * std::exp(U.solve((x.row(i) - y.row(j)).transpose()).squaredNorm() * -0.5);

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

                //                 double sf_d = 2 * AbstractKernel2<Params>::_sf2, sn_d = 2 * AbstractKernel2<Params>::_sn2,
                //                        k;

                //                 Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> U = tools::upperCholesky(_S);
                //                 Eigen::VectorXd prod;

                //                 size_t index;

                // #pragma omp parallel for collapse(2)
                //                 for (size_t j = 0; j < y_samples; j++)
                //                     for (size_t i = 0; i < x_samples; i++) {
                //                         index = j * x_samples + i;

                //                         k = std::exp(U.solve((x.row(i) - y.row(j)).transpose()).squaredNorm() * -0.5);
                //                         grad(index, 0) = sf_d * k;
                //                         grad(index, 1) = (j == i && &x == &y) ? sn_d : 0;

                //                         prod = U.transpose().solve(U.solve(x.row(i) - y.row(j)).transpose());
                //                         grad.row(index).segment(2, _S.size()) = (prod.transpose() * prod).reshaped(1, n_features) * k;
                //                     }

                return grad;
            }

            Eigen::MatrixXd hessianParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

        protected:
            Eigen::MatrixXd _S;

            /* Kernel */
            Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd k(x_samples, y_samples);

                // std::cout << _S << std::endl;

                Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> U = tools::upperCholesky(_S);

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        k(i, j) = AbstractKernel2<Params>::_sf2 * std::exp(U.solve((x.row(i) - y.row(j)).transpose()).squaredNorm() * -0.5)
                            + ((j == i && &x == &y) ? AbstractKernel2<Params>::_sn2 + 1e-8 : 0);

                return k;
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params)
            {
                AbstractKernel2<Params>::_params.segment(2, _S.size()) = params;
                _S = params.reshaped(_S.rows(), _S.cols());
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const { return _S.size(); }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNEL_LIB_KERNELS_SQUARED_EXP_FULL