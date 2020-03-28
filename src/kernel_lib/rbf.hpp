#ifndef KERNEL_LIB_RBF_HPP
#define KERNEL_LIB_RBF_HPP

#include "kernel_lib/abstract_kernel.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel_rbf {
            PARAM_SCALAR(double, sigma, 5);
        };
    } // namespace defaults

    enum class CovarianceType : unsigned int {
        SPHERICAL = 1 << 0,
        DIAGONAL = 1 << 1,
        FULL = 1 << 2,
    };

    template <typename Params>
    class Rbf : public AbstractKernel<Params, Rbf<Params>> {

        using Kernel_t = AbstractKernel<Params, Rbf<Params>>;

    public:
        Rbf(size_t dim = 1) : sigma_(Params::kernel_rbf::sigma()) {}

        Eigen::VectorXd kernel() const
        {
            return log_kernel().array().exp();
        }

        Eigen::VectorXd log_kernel() const
        {
            Eigen::VectorXd log_k(Kernel_t::x_samples_ * Kernel_t::y_samples_);

            size_t index = 0;

            for (size_t i = 0; i < Kernel_t::x_samples_; i++) {
                for (size_t j = 0; j < Kernel_t::y_samples_; j++) {
                    log_k(index) = (Kernel_t::x_.row(i) - Kernel_t::y_.row(j)).squaredNorm() * -0.5 / std::pow(sigma_, 2);
                    index++;
                }
            }

            return log_k;
        }

    protected:
        double sigma_;
    };

} // namespace kernel_lib

#endif // KERNEL_LIB_RBF_HPP