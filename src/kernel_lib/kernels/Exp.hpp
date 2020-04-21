#ifndef KERNELLIB_KERNELS_EXP_HPP
#define KERNELLIB_KERNELS_EXP_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include <Corrade/Containers/EnumSet.h>

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
        struct kernel_exp {
            // Settings
            PARAM_SCALAR(Covariance, type, CovarianceType::SPHERICAL);
            PARAM_SCALAR(bool, inverse, false);

            // Parameters
            PARAM_VECTOR(double, sigma, 5);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class Exp : public AbstractKernel<Params, Exp<Params>> {

            using Kernel_t = AbstractKernel<Params, Exp<Params>>;

        public:
            Exp() : _sigma(Params::kernel_exp::sigma()), _type(Params::kernel_exp::type()), _inverse(Params::kernel_exp::inverse())
            {
            }

            /* Parameters */
            Eigen::VectorXd parameters() const
            {
                Eigen::VectorXd params;

                return params;
            }

            void setParameters(const Eigen::VectorXd& params)
            {
            }

            Eigen::MatrixXd gradientParams() const
            {
                Eigen::MatrixXd grad_params;

                return grad_params;
            }

            /* Evaluate Kernel */
            Eigen::VectorXd kernel() const
            {
                return log_kernel().array().exp();
            }

            /* Evaluate Gradient */
            Eigen::MatrixXd gradient() const
            {
                Eigen::MatrixXd grad;

                return grad;
            }

            /* Evaluate Hessian */
            Eigen::MatrixXd hessian() const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

            /* Settings */
            Exp& setCovariance(const Covariance& cov)
            {
                _type = cov;

                return *this;
            }

            Exp& setInverse(const bool& inverse)
            {
                _inverse = inverse;

                return *this;
            }

        protected:
            Eigen::MatrixXd _sigma;

            Covariance _type;

            bool _inverse;

            Eigen::VectorXd log_kernel() const
            {
                Eigen::VectorXd log_k(Kernel_t::_x_samples * Kernel_t::_y_samples);
                size_t index = 0;

                if (_type & CovarianceType::SPHERICAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == 1, "Sigma requires dimension 1")

                    double sig = -0.5 / std::pow(_sigma(0, 0), 2);

                    for (size_t i = 0; i < Kernel_t::_x_samples; i++) {
                        for (size_t j = 0; j < Kernel_t::_y_samples; j++) {
                            log_k(index) = (Kernel_t::_x.row(i) - Kernel_t::_y.row(j)).squaredNorm() * sig;
                            index++;
                        }
                    }
                }
                else if (_type & CovarianceType::DIAGONAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == Kernel_t::_n_features, "Sigma requires dimension equal to the number of features")

                    Eigen::Array<double, 1, Eigen::Dynamic> sig = -0.5 * _sigma.transpose().array().pow(2).inverse();

                    for (size_t i = 0; i < Kernel_t::_x_samples; i++) {
                        for (size_t j = 0; j < Kernel_t::_y_samples; j++) {
                            log_k(index) = ((Kernel_t::_x.row(i) - Kernel_t::_y.row(j)).array().square() * sig).sum();
                            index++;
                        }
                    }
                }
                else if (_type & CovarianceType::FULL) {
                    REQUIRED_DIMENSION(_sigma.rows() == std::pow(Kernel_t::_n_features, 2), "Sigma requires dimension equal to squared number of features")

                    if (_inverse) {
                        const Eigen::MatrixXd& sig = _sigma.reshaped(Kernel_t::_n_features, Kernel_t::_n_features) * -0.5;

                        for (size_t i = 0; i < Kernel_t::_x_samples; i++) {
                            for (size_t j = 0; j < Kernel_t::_y_samples; j++) {
                                const Eigen::VectorXd& v = Kernel_t::_x.row(i) - Kernel_t::_y.row(j);
                                log_k(index) = v.transpose() * sig * v;
                                index++;
                            }
                        }
                    }
                    else {
                        // Why this? const and reference?
                        const Chol::Traits::MatrixL& L = tools::cholesky(_sigma.reshaped(Kernel_t::_n_features, Kernel_t::_n_features));

                        for (size_t i = 0; i < Kernel_t::_x_samples; i++) {
                            for (size_t j = 0; j < Kernel_t::_y_samples; j++) {
                                log_k(index) = L.solve((Kernel_t::_x.row(i) - Kernel_t::_y.row(j)).transpose()).squaredNorm() * -0.5;
                                index++;
                            }
                        }
                    }
                }

                return log_k;
            }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_EXP_HPP