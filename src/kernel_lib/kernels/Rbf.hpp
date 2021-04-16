#ifndef KERNELLIB_KERNELS_RBF_HPP
#define KERNELLIB_KERNELS_RBF_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include <Corrade/Containers/EnumSet.h>
#include <Corrade/Containers/Pointer.h>
// #include <tbb/tbb.h>

#define LOG_KERNEL(x, y) \
    (x - y).squaredNorm() * sig;

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
            Rbf() : _sigma(Params::rbf::sigma()), _type(Params::rbf::type())
            {
                _sig = -0.5 / std::pow(_sigma(0, 0), 2);
            }

            /* Settings */
            Rbf& setCovariance(const Covariance& cov)
            {
                _type = cov;

                return *this;
            }

            /**
             * @brief Initializes the FrankaHW class to be fully operational. This involves parsing required
             * configurations from the ROS parameter server, connecting to the robot and setting up interfaces
             * for the ros_control framework.
             *
             * @tparam example of template param
             * 
             * @param[in] param example of function input param
             * 
             * @note example of note
             *
             * @return True if successful, false otherwise.
             */
            Eigen::MatrixXd logKernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd log_k(x.rows(), y.rows());

                /* Isotropic Kernel */
                if (_type & CovarianceType::SPHERICAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == 1, "Sigma requires dimension 1")

#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y.rows(); j++)
                        for (size_t i = 0; i < x.rows(); i++)
                            log_k(i, j) = (x.row(i) - y.row(j)).squaredNorm() * _sig;
                }
                /* Diagonal Kernel */
                else if (_type & CovarianceType::DIAGONAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == x.cols(), "Sigma requires dimension equal to the number of features")

                    Eigen::Array<double, 1, Eigen::Dynamic> sig = -0.5 * _sigma.transpose().array().pow(2).inverse();

#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y.rows(); j++)
                        for (size_t i = 0; i < x.rows(); i++)
                            log_k(i, j) = exp(((x.row(i) - y.row(j)).array().square() * sig).sum());
                }
                /* Full Kernel */
                else if (_type & CovarianceType::FULL) {
                    REQUIRED_DIMENSION(_sigma.rows() == std::pow(x.cols(), 2), "Sigma requires dimension equal to squared number of features")

                    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> U = tools::upperCholesky(_sigma.reshaped(x.cols(), x.cols()));

#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y.rows(); j++)
                        for (size_t i = 0; i < x.rows(); i++)
                            log_k(i, j) = std::exp(U.solve((x.row(i) - y.row(j)).transpose()).squaredNorm() * -0.5);
                }

                return std::move(log_k);
            }

        protected:
            Eigen::MatrixXd _sigma;

            double _sig;

            Covariance _type;

            // inline double calc_log(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const
            // {
            //     return (x - y).squaredNorm() * _sig;
            // }

            Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const override
            {
                REQUIRED_DIMENSION(x.cols() == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd k = logKernel(x, y);

                k = k.array().exp(); // k.unaryExpr(&Exp)

                return std::move(k);
            }

            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const override
            {
                Eigen::MatrixXd grad = logKernelGradient(x, y);

                if (i == 1)
                    for (size_t j = 0; j < grad.cols(); j++) {
                        grad.col(j) = grad.col(j).array() * this->operator()(x, y).reshaped(grad.rows(), 1).array();
                    }

                return grad;
            }

            Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const override
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

        public:
            Eigen::MatrixXd logKernelGradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd eval(x_samples * y_samples, n_features);

                /* Isotropic Kernel */
                if (_type & CovarianceType::SPHERICAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == 1, "Sigma requires dimension 1")

                    double sig = 1 / std::pow(_sigma(0, 0), 2);

#pragma omp parallel for collapse(2)
                    for (size_t i = 0; i < x.rows(); i++)
                        for (size_t j = 0; j < y.rows(); j++)
                            eval.row(j * x_samples + i) = (x.row(i) - y.row(j)) * sig;
                }
                /* Diagonal Kernel */
                else if (_type & CovarianceType::DIAGONAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == x.cols(), "Sigma requires dimension equal to the number of features")

                    Eigen::Array<double, 1, Eigen::Dynamic> sig = _sigma.transpose().array().pow(2).inverse();

#pragma omp parallel for collapse(2)
                    for (size_t i = 0; i < x.rows(); i++)
                        for (size_t j = 0; j < y.rows(); j++)
                            eval.row(i * y_samples + j) = (x.row(i) - y.row(j)).array() * sig;
                }
                /* Full Kernel */
                else if (_type & CovarianceType::FULL) {
                    REQUIRED_DIMENSION(_sigma.rows() == std::pow(x.cols(), 2), "Sigma requires dimension equal to squared number of features")

                    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> U = tools::upperCholesky(_sigma.reshaped(x.cols(), x.cols()));

#pragma omp parallel for collapse(2)
                    for (size_t i = 0; i < x.rows(); i++)
                        for (size_t j = 0; j < y.rows(); j++)
                            eval.row(i * y_samples + j) = U.transpose().solve(U.solve((x.row(i) - y.row(j)).transpose()));
                }

                return std::move(eval);
            }

            Eigen::MatrixXd logKernelGradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows(), n_features = x.cols();

                REQUIRED_DIMENSION(n_features == y.cols(), "Y must have the same dimension of X")

                Eigen::MatrixXd eval(x_samples * y_samples, _sigma.size());

                /* Isotropic Kernel */
                if (_type & CovarianceType::SPHERICAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == 1, "Sigma requires dimension 1")

                    double sig = 1 / std::pow(_sigma(0, 0), 3);

#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y.rows(); j++)
                        for (size_t i = 0; i < x.rows(); i++)
                            eval.row(i * y_samples + j) = (x.row(i) - y.row(j)).squaredNorm() * sig;
                }
                /* Diagonal Kernel */
                else if (_type & CovarianceType::DIAGONAL) {
                    REQUIRED_DIMENSION(_sigma.rows() == x.cols(), "Sigma requires dimension equal to the number of features")

                    Eigen::Array<double, 1, Eigen::Dynamic> sig = _sigma.transpose().array().pow(3).inverse();

#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y.rows(); j++)
                        for (size_t i = 0; i < x.rows(); i++)
                            eval.row(i * y_samples + j) = (x.row(i) - y.row(j)).array().square() * sig;
                }
                /* Full Kernel */
                else if (_type & CovarianceType::FULL) {
                    REQUIRED_DIMENSION(_sigma.rows() == std::pow(x.cols(), 2), "Sigma requires dimension equal to squared number of features")

                    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> U = tools::upperCholesky(_sigma.reshaped(x.cols(), x.cols()));

#pragma omp parallel for collapse(2)
                    for (size_t j = 0; j < y.rows(); j++)
                        for (size_t i = 0; i < x.rows(); i++) {
                            Eigen::VectorXd prod = U.transpose().solve(U.solve(x.row(i) - y.row(j)).transpose());
                            eval.row(i * y_samples + j) = (prod.transpose() * prod).reshaped(1, n_features);
                        }
                }

                return std::move(eval);
            }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_RBF_HPP