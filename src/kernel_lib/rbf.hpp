#ifndef KERNEL_LIB_RBF_HPP
#define KERNEL_LIB_RBF_HPP

#include "kernel_lib/abstract_kernel.hpp"

namespace kernel_lib {
    namespace defaults {
        struct kernel_rbf {
            BO_PARAM(double, sigma, 1);
        };
    } // namespace defaults

    template <typename Params>
    class Rbf : public AbstractKernel<Params, Rbf<Params>> {
    public:
        Rbf(size_t dim = 1) : sigma_(Params::kernel_rbf::sigma()) {}

        Eigen::MatrixXd kernel() const
        {
            Eigen::MatrixXd log_k = log_kernel();

            return log_k.array().exp();
        }

        Eigen::MatrixXd log_kernel() const
        {
            Eigen::MatrixXd log_k = -AbstractKernel<Params, Rbf<Params>>::diff_.rowwise().squaredNorm() * sigma_ / 2;
            return log_k;
        }

    protected:
        double sigma_;
    };

} // namespace kernel_lib

#endif // KERNEL_LIB_RBF_HPP