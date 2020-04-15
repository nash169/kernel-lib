#ifndef KERNELLIB_ABSTRACTKERNEL_HPP
#define KERNELLIB_ABSTRACTKERNEL_HPP

#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

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
        AbstractKernel() : _sigma_f(Params::kernel::sigma_f())
        {
            _sigma_n = (Params::kernel::sigma_n() == 0) ? 1e-8 : Params::kernel::sigma_n();
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
            _n_features = x.cols();
            REQUIRED_DIMENSION(_n_features == y.cols(), "Y must have the same dimension of X")

            _x_samples = x.rows();
            _y_samples = y.rows();

            _x = x;
            _y = y;
        }

    protected:
        size_t _n_features, _x_samples, _y_samples;

        double _sigma_f, _sigma_n;

        Eigen::MatrixXd _x, _y;
    };

} // namespace kernel_lib

#endif // KERNELLIB_ABSTRACTKERNEL_HPP