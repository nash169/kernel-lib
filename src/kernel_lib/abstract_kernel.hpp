#ifndef KERNEL_LIB_KERNEL_HPP
#define KERNEL_LIB_KERNEL_HPP

#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"
#include <Eigen/Core>

namespace kernel_lib {
    namespace defaults {
        struct kernel {
            BO_PARAM(double, sigma_f, 1.0);
            BO_PARAM(double, sigma_n, 0.0);
        };
    } // namespace defaults

    template <typename Params, typename Kernel>
    class AbstractKernel {
    public:
        AbstractKernel(size_t dim = 1) : sigma_f_(Params::kernel::sigma_f())
        {
            sigma_n_ = (Params::kernel::sigma_n() == 0) ? 1e-8 : Params::kernel::sigma_n();
        }

        Eigen::MatrixXd operator()()
        {
            Eigen::Matrix<bool, Eigen::Dynamic, 1> vec_noise = diff_.rowwise().norm().array() <= 1e-8;

            return static_cast<const Kernel*>(this)->kernel() * sigma_f_ + vec_noise.cast<double>() * sigma_n_;
        }

        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
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
            X_ = x.replicate(y_samples_, 1);
            // Y_ = tools::repeat(y, x_samples_, 1);
            Y_ = y(Eigen::VectorXi::LinSpaced(x_samples_ * y_samples_, 0, y_samples_ - 1), Eigen::all);
            diff_ = X_ - Y_;
        }

    protected:
        int n_features_, x_samples_, y_samples_;

        double sigma_f_, sigma_n_;

        Eigen::MatrixXd x_, y_, X_, Y_, diff_;
    };

} // namespace kernel_lib

#endif // KERNEL_LIB_KERNEL_HPP