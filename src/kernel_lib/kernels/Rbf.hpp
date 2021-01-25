#ifndef KERNELLIB_KERNELS_RBF_HPP
#define KERNELLIB_KERNELS_RBF_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include <Corrade/Containers/EnumSet.h>
#include <Corrade/Containers/Pointer.h>
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

            Corrade::Containers::Pointer<Eigen::MatrixXd> logKernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Corrade::Containers::Pointer<Eigen::MatrixXd> k(new Eigen::MatrixXd(x.rows(), y.rows()));

                if (_type & CovarianceType::SPHERICAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == 1, "Sigma requires dimension 1")

                    double sig = -0.5 / std::pow(_sigma(0, 0), 2);

#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y.rows(); j++)
                        for (size_t i = 0; i < x.rows(); i++)
                            (*k)(i, j) = (x.row(i) - y.row(j)).squaredNorm() * sig;
                }
                else if (_type & CovarianceType::DIAGONAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == x.cols(), "Sigma requires dimension equal to the number of features")

                    Eigen::Array<double, 1, Eigen::Dynamic> sig = -0.5 * _sigma.transpose().array().pow(2).inverse();

#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y.rows(); j++)
                        for (size_t i = 0; i < x.rows(); i++)
                            (*k)(i, j) = exp(((x.row(i) - y.row(j)).array().square() * sig).sum());
                }
                else if (_type & CovarianceType::FULL) {
                    REQUIRED_DIMENSION(_sigma.rows() == std::pow(x.cols(), 2), "Sigma requires dimension equal to squared number of features")

                    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> U = tools::upperCholesky(_sigma.reshaped(x.cols(), x.cols()));

#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y.rows(); j++)
                        for (size_t i = 0; i < x.rows(); i++)
                            (*k)(i, j) = std::exp(U.solve((x.row(i) - y.row(j)).transpose()).squaredNorm() * -0.5);
                }

                return std::move(k);
            }

        protected:
            Eigen::MatrixXd _sigma;

            Covariance _type;

            bool _inverse;

            Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                REQUIRED_DIMENSION(x.cols() == y.cols(), "Y must have the same dimension of X")

                // logKernel(k, x, y);

                Corrade::Containers::Pointer<Eigen::MatrixXd> k = logKernel(x, y);

                k->array() = k->array().exp(); // k.unaryExpr(&Exp)

                return std::move(*k);
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