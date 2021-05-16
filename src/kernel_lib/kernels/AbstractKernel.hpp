#ifndef KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP
#define KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP

#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {

    namespace defaults {
        struct kernel {
            // Log signal std
            PARAM_SCALAR(double, sf, 1.0);

            // Log noise std
            PARAM_SCALAR(double, sn, 0.0);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params, typename Kernel>
        class AbstractKernel {
        public:
            AbstractKernel() : _sf2(std::exp(2 * Params::kernel::sf())), _sn2(std::exp(2 * Params::kernel::sn())) {}

            /* Evaluate kernel */
            template <typename Derived>
            inline __attribute__((always_inline)) double operator()(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                return _sf2 * static_cast<const Kernel*>(this)->kernel(x, y) + ((&x == &y) ? _sn2 + 1e-8 : 0);
            }

            template <typename Derived>
            inline __attribute__((always_inline)) auto grad(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                return _sf2 * static_cast<const Kernel*>(this)->gradient(x, y);
            }

            template <typename Derived>
            inline __attribute__((always_inline)) auto gradParams(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y) const
            {
                Eigen::VectorXd p(sizeParams());
                p << 2 * _sf2 * static_cast<const Kernel*>(this)->kernel(x, y), (&x == &y) ? 2 * _sf2 : 0, _sn2 * static_cast<const Kernel*>(this)->gradientParams(x, y);

                return p;
            }

            template <int Size>
            Eigen::MatrixXd gramian(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x, const Eigen::Matrix<double, Eigen::Dynamic, Size>& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows();

                Eigen::MatrixXd k(x_samples, y_samples);

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        k(i, j) = (*this)(x.row(i), y.row(j));

                return k;
            }

            template <int Size>
            Eigen::Matrix<double, Eigen::Dynamic, Size> multiGrad(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x, const Eigen::Matrix<double, Eigen::Dynamic, Size>& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows();

                Eigen::MatrixXd g(x_samples * y_samples, x.cols());

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        g.row(j * x_samples + i) = grad(x.row(i), y.row(j));

                return g;
            }

            template <int Size>
            Eigen::MatrixXd multiGradParams(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x, const Eigen::Matrix<double, Eigen::Dynamic, Size>& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows();

                Eigen::MatrixXd p(x_samples * y_samples, sizeParams());

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        p.row(j * x_samples + i) = gradParams(x.row(i), y.row(j));

                return p;
            }

            /* Parameters */
            Eigen::VectorXd params() const
            {
                Eigen::VectorXd params(this->sizeParams());
                params << std::log(_sf2) / 2, std::log(_sn2) / 2, this->parameters();

                return params;
            }

            /* Set parameters */
            void setParams(const Eigen::VectorXd& params)
            {
                _sf2 = std::exp(2 * params(0));
                _sn2 = std::exp(2 * params(1));

                this->setParameters(params.segment(2, params.rows() - 2));
            }

            /* Parameters' size */
            size_t sizeParams() const { return this->sizeParameters() + 2; }

        protected:
            double _sf2, _sn2;

            /* Get specific kernel parameters */
            virtual Eigen::VectorXd parameters() const = 0;

            /* Set specific kernel parameters */
            virtual void setParameters(const Eigen::VectorXd& params) = 0;

            /* Get number of parameters for the specific kernel */
            virtual size_t sizeParameters() const = 0;
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP