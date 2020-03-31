#ifndef KERNEL_LIB_RBF_HPP
#define KERNEL_LIB_RBF_HPP

#include "kernel_lib/AbstractKernel.hpp"
#include <Corrade/Containers/EnumSet.h>

using Chol = Eigen::LLT<Eigen::MatrixXd>;

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
        struct kernel_rbf {
            PARAM_SCALAR(Covariance, type, CovarianceType::SPHERICAL);
            PARAM_SCALAR(bool, inverse, false);
            PARAM_VECTOR(double, sigma, 5);
        };
    } // namespace defaults

    namespace kernel {
        template <typename Params>
        class Rbf : public AbstractKernel<Params, Rbf<Params>> {

            using Kernel_t = AbstractKernel<Params, Rbf<Params>>;

        public:
            Rbf() : _sigma(Params::kernel_rbf::sigma()), _type(Params::kernel_rbf::type())
            {
            }

            Eigen::VectorXd kernel() const
            {
                return log_kernel().array().exp();
            }

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
                        Eigen::MatrixXd sig = _sigma.reshaped(Kernel_t::_n_features, Kernel_t::_n_features) * -0.5;
                        for (size_t i = 0; i < Kernel_t::_x_samples; i++) {
                            for (size_t j = 0; j < Kernel_t::_y_samples; j++) {
                                Eigen::VectorXd v = (Kernel_t::_x.row(i) - Kernel_t::_x.row(j));
                                log_k(index) = v.transpose() * sig * v;
                                index++;
                            }
                        }
                    }
                    else {
                        // Chol::Traits::MatrixL L = cholesky();

                        // for (size_t i = 0; i < Kernel_t::_x_samples; i++) {
                        //     for (size_t j = 0; j < Kernel_t::_y_samples; j++) {
                        //         log_k(index) = L.solve(Kernel_t::_x.row(i) - Kernel_t::_x.row(j)).squaredNorm() * -0.5;
                        //         index++;
                        //     }
                        // }
                    }
                }

                return log_k;
            }

        protected:
            Eigen::MatrixXd _sigma;

            Covariance _type;

            bool _inverse;

            // Code from limbo to calculate the cholesky (check if optimized)
            // also check if it is ok returning inside the is statement
            // with this thing compilation time gets super long why?
            Chol::Traits::MatrixL cholesky() const
            {
                Chol chol(_sigma);

                if (chol.info() != Eigen::Success) {
                    // There was an error; probably the matrix is not SPD
                    // Let's try to make it SPD and take cholesky of that
                    // original MATLAB code: http://fr.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
                    // Note that at this point _L is not cholesky factor, but matrix to be factored

                    // Symmetrize A into B
                    Eigen::MatrixXd B = (_sigma.array() + _sigma.transpose().array()) / 2.;

                    // Compute the symmetric polar factor of B. Call it H. Clearly H is itself SPD.
                    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    Eigen::MatrixXd V, Sigma, H, L_hat;

                    Sigma = Eigen::MatrixXd::Identity(B.rows(), B.cols());
                    Sigma.diagonal() = svd.singularValues();
                    V = svd.matrixV();

                    H = V * Sigma * V.transpose();

                    // Get candidate for closest SPD matrix to _sigma
                    L_hat = (B.array() + H.array()) / 2.;

                    // Ensure symmetry
                    L_hat = (L_hat.array() + L_hat.array()) / 2.;

                    // Test that L_hat is in fact PD. if it is not so, then tweak it just a bit.
                    Eigen::LLT<Eigen::MatrixXd> llt_hat(L_hat);
                    int k = 0;
                    while (llt_hat.info() != Eigen::Success) {
                        k++;
                        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(L_hat);
                        double min_eig = es.eigenvalues().minCoeff();
                        L_hat.diagonal().array() += (-min_eig * k * k + 1e-50);
                        llt_hat.compute(L_hat);
                    }

                    return llt_hat.matrixL();
                }
                else {
                    return chol.matrixL();
                }
            }
        };
    } // namespace kernel
} // namespace kernel_lib

#endif // KERNEL_LIB_RBF_HPP