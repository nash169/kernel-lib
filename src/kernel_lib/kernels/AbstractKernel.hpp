#ifndef KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP
#define KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP

#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"
#include "kernel_lib/utils/Expansion.hpp"

namespace kernel_lib {

    namespace defaults {
        struct kernel {
            // Log signal std
            PARAM_SCALAR(double, sf, 0.0);

            // Log noise std
            PARAM_SCALAR(double, sn, -1e2);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params, typename Kernel>
        class AbstractKernel {
        public:
            AbstractKernel() : _sf2(std::exp(2 * Params::kernel::sf())), _sn2(std::exp(2 * Params::kernel::sn())) {}

            /* Kernel */
            // Following what said below for kernelImpl we use MatrixBase type to automatically derived the size to of the vector
            // https://stackoverflow.com/questions/25094948/eigen-how-to-access-the-underlying-array-of-a-matrixbasederived
            // For the usage of MatrixBase: https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
            template <typename Derived, typename OtherDerived>
            EIGEN_ALWAYS_INLINE double operator()(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<OtherDerived>& y) const
            {
                return kernelImpl < (Derived::RowsAtCompileTime == 1) ? Derived::ColsAtCompileTime : Derived::RowsAtCompileTime > (x.derived(), y.derived());
            }

            /* Gradient */
            template <typename Derived, typename OtherDerived>
            EIGEN_ALWAYS_INLINE auto grad(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<OtherDerived>& y, const size_t& i = 1) const
            {
                return gradImpl < (Derived::RowsAtCompileTime == 1) ? Derived::ColsAtCompileTime : Derived::RowsAtCompileTime > (x.derived(), y.derived(), i);
            }

            /* Parameters' gradient */
            template <typename Derived, typename OtherDerived>
            EIGEN_ALWAYS_INLINE auto gradParams(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<OtherDerived>& y) const
            {
                return gradParamsImpl < (Derived::RowsAtCompileTime == 1) ? Derived::ColsAtCompileTime : Derived::RowsAtCompileTime > (x.derived(), y.derived());
            }

            /* Gram matrix */
            // Flattening not really necessary here
            // https://awesomekling.github.io/Smarter-C++-inlining-with-attribute-flatten/
            template <int Size>
            EIGEN_DEVICE_FUNC Eigen::MatrixXd gram(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x, const Eigen::Matrix<double, Eigen::Dynamic, Size>& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows();

                Eigen::MatrixXd k(x_samples, y_samples);

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        k(i, j) = kernelImpl<Size>(x.row(i), y.row(j));

                return k;
            }

            /* Gram matrix gradient */
            template <int Size>
            Eigen::Matrix<double, Eigen::Dynamic, Size> gramGrad(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x, const Eigen::Matrix<double, Eigen::Dynamic, Size>& y, const size_t& k = 1) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows();

                Eigen::MatrixXd g(x_samples * y_samples, x.cols());

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        g.row(j * x_samples + i) = gradImpl<Size>(x.row(i), y.row(j), k);

                return g;
            }

            /* Gram matrix gradient with respect to parameters */
            template <int Size>
            Eigen::MatrixXd gramGradParams(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x, const Eigen::Matrix<double, Eigen::Dynamic, Size>& y) const
            {
                size_t x_samples = x.rows(), y_samples = y.rows();

                Eigen::MatrixXd p(x_samples * y_samples, sizeParams());

#pragma omp parallel for collapse(2)
                for (size_t j = 0; j < y_samples; j++)
                    for (size_t i = 0; i < x_samples; i++)
                        p.row(j * x_samples + i) = gradParamsImpl<Size>(x.row(i), y.row(j));

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

            friend class utils::Expansion<Params, Kernel>;

        protected:
            double _sf2, _sn2;

            /* Get specific kernel parameters */
            virtual Eigen::VectorXd parameters() const = 0;

            /* Set specific kernel parameters */
            virtual void setParameters(const Eigen::VectorXd& params) = 0;

            /* Get number of parameters for the specific kernel */
            virtual size_t sizeParameters() const = 0;

            /* Kernel Implementation */
            // Automatic template deduction fails if this is exposed to the user
            // https://stackoverflow.com/questions/48511569/template-argument-deduction-for-eigenrefmatt
            // For the usage of Eigen::Ref: https://eigen.tuxfamily.org/dox/classEigen_1_1Ref.html
            // For the usage of Eigen::Map: https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html#TutorialMapTypes
            // Setting the stride: https://eigen.tuxfamily.org/dox/classEigen_1_1Stride.html
            //                     https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
            template <int Size>
            EIGEN_ALWAYS_INLINE double kernelImpl(const Eigen::Ref<const Eigen::Matrix<double, Size, 1>, 0, Eigen::InnerStride<>>& x, const Eigen::Ref<const Eigen::Matrix<double, Size, 1>, 0, Eigen::InnerStride<>>& y) const
            {
                return _sf2 * static_cast<const Kernel*>(this)->kernel(x, y) + (x.data() == y.data() ? _sn2 + 1e-8 : 0);
            }

            /* Gradient Implementation */
            template <int Size>
            EIGEN_ALWAYS_INLINE auto gradImpl(const Eigen::Ref<const Eigen::Matrix<double, Size, 1>, 0, Eigen::InnerStride<>>& x, const Eigen::Ref<const Eigen::Matrix<double, Size, 1>, 0, Eigen::InnerStride<>>& y, const size_t& i = 1) const
            {
                return _sf2 * static_cast<const Kernel*>(this)->gradient(x, y, i);
            }

            /* Gradient parameters implementation */
            template <int Size>
            EIGEN_ALWAYS_INLINE auto gradParamsImpl(const Eigen::Ref<const Eigen::Matrix<double, Size, 1>, 0, Eigen::InnerStride<>>& x, const Eigen::Ref<const Eigen::Matrix<double, Size, 1>, 0, Eigen::InnerStride<>>& y) const
            {
                Eigen::VectorXd p(sizeParams());

                p << 2 * _sf2 * static_cast<const Kernel*>(this)->kernel(x, y),
                    (x.data() == y.data()) ? 2 * _sn2 : 0,
                    _sf2 * static_cast<const Kernel*>(this)->gradientParams(x, y);

                return p;
            }
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_ABSTRACTKERNEL_HPP

/* Alternative operator implementation that forces specific shape of the data */
// template <int Size>
// EIGEN_ALWAYS_INLINE double operator()(const Eigen::Matrix<double, Size, 1>& x, const Eigen::Matrix<double, Size, 1>& y) const
// {
//     return kernelImpl<Size>(x, y);
// }