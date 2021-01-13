#ifndef KERNELLIB_KERNELS_RBF_HPP
#define KERNELLIB_KERNELS_RBF_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include <Corrade/Containers/EnumSet.h>
// #include <tbb/tbb.h>

namespace kernel_lib {
    enum class CovarianceType : unsigned int {
        SPHERICAL = 1 << 0,
        DIAGONAL = 1 << 1,
        FULL = 1 << 2,
    };

    using Covariance = Corrade::Containers::EnumSet<CovarianceType>;
    CORRADE_ENUMSET_OPERATORS(Covariance)

    namespace defaults {
        struct rbf {
            // Settings
            PARAM_SCALAR(Covariance, type, CovarianceType::SPHERICAL);

            // Parameters
            PARAM_VECTOR(double, sigma, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class Rbf : public AbstractKernel<Params> {
        public:
            Rbf() : _sigma(Params::rbf::sigma()), _type(Params::rbf::type()) {}

            /* Settings */
            Rbf& setCovariance(const Covariance& cov)
            {
                _type = cov;

                return *this;
            }

            void logKernel(Eigen::MatrixXd& k, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                if (_type & CovarianceType::SPHERICAL) {
                    spherical(k, x, y);
                }
                else if (_type & CovarianceType::DIAGONAL) {
                    diagonal(k, x, y);
                }
                else if (_type & CovarianceType::FULL) {
                    full(k, x, y);
                }
            }

        protected:
            Eigen::MatrixXd _sigma;

            Covariance _type;

            bool _inverse;

            Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                REQUIRED_DIMENSION(x.cols() == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd k(x.rows(), y.rows());

                double sf2 = std::exp(2 * std::log(AbstractKernel<Params>::_sigma_f)), sn2 = std::exp(2 * std::log(AbstractKernel<Params>::_sigma_n));

                logKernel(k, x, y);

                k = sf2 * k.array().exp();

                k.diagonal().array() += sn2;

                return k;
            }

            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                Eigen::MatrixXd grad;

                return grad;
            }

            Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                Eigen::MatrixXd hess;

                return hess;
            }

            Eigen::VectorXd parameters() const override
            {
                return _sigma;
            }

            void setParameters(const Eigen::VectorXd& params) override
            {
                _sigma = params;
            }

            Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                Eigen::MatrixXd grad_params;

                return grad_params;
            }

            size_t sizeParameters() const override
            {
                return _sigma.rows();
            }

        private:
            /* Spherical (Isotropic) RBF Kernel*/
            void spherical(Eigen::MatrixXd& k, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                REQUIRED_DIMENSION(_sigma.rows() == 1, "Sigma requires dimension 1")

                double sig = -0.5 / std::pow(_sigma(0, 0), 2);

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y.rows(); j++)
                    for (size_t i = 0; i < x.rows(); i++)
                        k(i, j) = (x.row(i) - y.row(j)).squaredNorm() * sig;
            }

            /* Diagonal (ARD) RBF Kernel*/
            void diagonal(Eigen::MatrixXd& k, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                REQUIRED_DIMENSION(_sigma.rows() == x.cols(), "Sigma requires dimension equal to the number of features")

                Eigen::Array<double, 1, Eigen::Dynamic> sig = -0.5 * _sigma.transpose().array().pow(2).inverse();

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y.rows(); j++)
                    for (size_t i = 0; i < x.rows(); i++)
                        k(i, j) = exp(((x.row(i) - y.row(j)).array().square() * sig).sum());
            }

            /* Full RBF Kernel*/
            void full(Eigen::MatrixXd& k, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                REQUIRED_DIMENSION(_sigma.rows() == std::pow(x.cols(), 2), "Sigma requires dimension equal to squared number of features")

                Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> U = tools::upperCholesky(_sigma.reshaped(x.cols(), x.cols()));

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y.rows(); j++)
                    for (size_t i = 0; i < x.rows(); i++)
                        k(i, j) = std::exp(U.solve((x.row(i) - y.row(j)).transpose()).squaredNorm() * -0.5);
            }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_RBF_HPP