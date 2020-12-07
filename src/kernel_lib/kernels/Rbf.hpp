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

        FIRST = 1 << 3,
        SECOND = 1 << 4
    };

    using Covariance = Corrade::Containers::EnumSet<CovarianceType>;
    CORRADE_ENUMSET_OPERATORS(Covariance)

    namespace defaults {
        struct rbf {
            // Settings
            PARAM_SCALAR(Covariance, type, CovarianceType::SPHERICAL);
            PARAM_SCALAR(bool, inverse, false);

            // Parameters
            PARAM_VECTOR(double, sigma, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class Rbf : public AbstractKernel<Params> {
        public:
            Rbf() : _sigma(Params::rbf::sigma()), _type(Params::rbf::type()), _inverse(Params::rbf::inverse()) {}

            /* Settings */
            Rbf& setCovariance(const Covariance& cov)
            {
                _type = cov;

                return *this;
            }

            Rbf& setInverse(const bool& inverse)
            {
                _inverse = inverse;

                return *this;
            }

        protected:
            Eigen::MatrixXd _sigma;

            Covariance _type;

            bool _inverse;

            Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();
                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd k(x_samples, y_samples);

                if (_type & CovarianceType::SPHERICAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == 1, "Sigma requires dimension 1")

                    double sig = -0.5 / std::pow(_sigma(0, 0), 2);

#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y_samples; j++)
                        for (size_t i = 0; i < x_samples; i++) {
                            k(i, j) = exp((x.row(i) - y.row(j)).squaredNorm() * sig) * pow(AbstractKernel<Params>::_sigma_f, 2) + ((j == i) ? pow(AbstractKernel<Params>::_sigma_n, 2) + 1e-8 : 0);
                        }
                }
                else if (_type & CovarianceType::DIAGONAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == n_features, "Sigma requires dimension equal to the number of features")

                    Eigen::Array<double, 1, Eigen::Dynamic> sig = -0.5 * _sigma.transpose().array().pow(2).inverse();

#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y_samples; j++)
                        for (size_t i = 0; i < x_samples; i++) {
                            k(i, j) = exp(((x.row(i) - y.row(j)).array().square() * sig).sum()) * pow(AbstractKernel<Params>::_sigma_f, 2) + ((j == i) ? pow(AbstractKernel<Params>::_sigma_n, 2) + 1e-8 : 0);
                        }
                }
                else if (_type & CovarianceType::FULL) {
                    REQUIRED_DIMENSION(_sigma.rows() == std::pow(n_features, 2), "Sigma requires dimension equal to squared number of features")

                    if (_inverse) {
                        const Eigen::MatrixXd& sig = _sigma.reshaped(n_features, n_features) * -0.5;

#pragma omp parallel for collapse(2)
                        for (size_t j = 0; j < y_samples; j++) {
                            for (size_t i = 0; i < x_samples; i++) {
                                Eigen::VectorXd v = x.row(i) - y.row(j);
                                k(i, j) = exp(v.transpose() * sig * v) * pow(AbstractKernel<Params>::_sigma_f, 2) + ((j == i) ? pow(AbstractKernel<Params>::_sigma_n, 2) + 1e-8 : 0);
                            }
                        }
                    }
                    else {
                        const Chol::Traits::MatrixL& L = tools::cholesky(_sigma.reshaped(n_features, n_features));

#pragma omp parallel for collapse(2)
                        for (size_t j = 0; j < y_samples; j++)
                            for (size_t i = 0; i < x_samples; i++) {
                                // k(i, j) = exp(L.solve((x.row(i) - y.row(j)).transpose()).squaredNorm() * -0.5) * AbstractKernel<Params>::_sigma_f + ((j == i) ? AbstractKernel<Params>::_sigma_n + 1e-8 : 0);
                                k(i, j) = exp(L.solve((x.row(i) - y.row(j)).transpose()).squaredNorm());
                            }
                    }
                }

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
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_RBF_HPP