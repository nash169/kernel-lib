#ifndef KERNEL_LIB_SQUARED_EXP_FULL_HPP
#define KERNEL_LIB_SQUARED_EXP_FULL_HPP

#include <memory>

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/helper.hpp"

namespace kernel_lib {
    namespace defaults {
        struct exp_sq_full {
            // Log length
            PARAM_VECTOR(double, S, 1, 0.5, 0.5, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class SquaredExpFull : public AbstractKernel<Params, SquaredExpFull<Params>> {
        public:
            SquaredExpFull() : _S(Params::exp_sq_full::S())
            {
                int size = std::sqrt(Params::exp_sq_full::S().size());

                _S = Eigen::Map<Eigen::MatrixXd>(const_cast<double*>(Params::exp_sq_full::S().data()), size, size);

                // Check if this is using LAPACK
                // Integrate with tools::cholesky()
                _llt = std::make_unique<Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>, Eigen::Lower>>(_S);
            }

            /* Overload kernel for handling single sample evaluation */
            template <typename Derived>
            EIGEN_ALWAYS_INLINE double kernel(const Derived& x, const Derived& y) const
            {
                return std::exp(_llt->matrixL().solve(x - y).squaredNorm() * -0.5);
            }

            /* Overload gradient for handling single sample evaluation (fastest solution but there is some issue for inferring size) */
            template <typename Derived>
            EIGEN_ALWAYS_INLINE auto gradient(const Derived& x, const Derived& y, const size_t& i = 1) const
            {
                return _llt->solve(i ? (y - x) : (x - y)) * kernel(x, y);
            }

            template <typename Derived>
            EIGEN_ALWAYS_INLINE auto gradientParams(const Derived& x, const Derived& y) const
            {
                Eigen::MatrixXd s = (_llt->solve(x - y) * _llt->solve(x - y).transpose()) * kernel(x, y);

                return Eigen::Map<Eigen::VectorXd>(s.data(), s.size());
            }

        protected:
            // Full matrix (probably the same information of llt)
            Eigen::MatrixXd _S;

            // This should not allocate additional memory
            std::unique_ptr<Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>, Eigen::Lower>> _llt;

            Eigen::VectorXd parameters() const override
            {
                return Eigen::Map<Eigen::VectorXd>(const_cast<double*>(_S.data()), _S.size());
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params) override
            {
                // For the moment with receive all the entries of the covariance matrix;
                // for real optimization it is better to set directly the lower/upper triangular matrix.
                // Reminder: const_cast<Eigen::MatrixXd&>(llt.matrixLLT()).coeffRef(i, j): https://eigen.tuxfamily.org/bz/show_bug.cgi?id=1599
                // this does not require Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>>.
                // For sparse solution: https://forum.kde.org/viewtopic.php?f=74&t=128080
                int size = std::sqrt(params.size());

                _S = Eigen::Map<Eigen::MatrixXd>(const_cast<double*>(params.data()), size, size);

                _llt->compute(_S);
            }

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const override { return _S.size(); }
        };
    } // namespace kernels
} // namespace kernel_lib

#endif // KERNEL_LIB_SQUARED_EXP_FULL_HPP