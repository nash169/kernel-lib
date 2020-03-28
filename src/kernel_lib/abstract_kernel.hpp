#ifndef KERNEL_LIB_KERNEL_HPP
#define KERNEL_LIB_KERNEL_HPP

#include "kernel_lib/tools/Timer.hpp"
#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"
#include <Eigen/Dense>

namespace kernel_lib {
    namespace defaults {
        struct kernel {
            PARAM_SCALAR(double, sigma_f, 1.0);
            PARAM_SCALAR(double, sigma_n, 0.0);
        };
    } // namespace defaults

    template <typename Params, typename Kernel>
    class AbstractKernel {
    public:
        AbstractKernel(size_t dim = 1) : sigma_f_(Params::kernel::sigma_f())
        {
            sigma_n_ = (Params::kernel::sigma_n() == 0) ? 1e-8 : Params::kernel::sigma_n();
        }

        Eigen::VectorXd operator()() const
        {
            return static_cast<const Kernel*>(this)->kernel();
        }

        Eigen::VectorXd operator()(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
        {
            set_data(x, y);

            return (*this)();
        }

        void set_data(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
        {
            n_features_ = x.cols();
            assert(n_features_ == y.cols());

            x_samples_ = x.rows();
            y_samples_ = y.rows();

            x_ = x;
            y_ = y;
        }

    protected:
        int n_features_, x_samples_, y_samples_;

        double sigma_f_, sigma_n_;

        Eigen::MatrixXd x_, y_;
    };

} // namespace kernel_lib

#endif // KERNEL_LIB_KERNEL_HPP