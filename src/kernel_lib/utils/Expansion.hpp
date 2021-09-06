#ifndef KERNELLIB_EXPANSION_UTILS_HPP
#define KERNELLIB_EXPANSION_UTILS_HPP

#include "kernel_lib/tools/type_name_rt.hpp"

namespace kernel_lib {
    namespace defaults {
        struct expansion {
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class SquaredExp;

        template <typename Params, typename Function>
        class RiemannSqExp;
    } // namespace kernels

    namespace utils {
        template <typename Params, typename Kernel>
        class Expansion {
        public:
            Expansion() : _k() {}

            template <int Size>
            EIGEN_ALWAYS_INLINE double operator()(const Eigen::Matrix<double, Size, 1>& x) const
            {
                double r = 0;

                // Can I parallelize this?
                // https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-data.html
                for (size_t i = 0; i < _x.rows(); i++)
                    r += _w(i) * _k.template kernelImpl<Size>(_x.row(i), x);

                return r;
            }

            template <int Size>
            Eigen::VectorXd multiEval(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x) const
            {
                Eigen::VectorXd r(x.rows());

#pragma omp parallel for
                for (size_t i = 0; i < r.rows(); i++)
                    r(i) = this->operator()<Size>(x.row(i));

                return r;
            }

            template <int Size>
            Eigen::VectorXd multiEval2(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x) const
            {
                return _k.gram(_x, x).transpose() * _w;
            }

            // Check https://stackoverflow.com/questions/47035541/specialize-only-a-part-of-one-method-of-a-template-class
            //       https://stackoverflow.com/questions/12877420/specialization-of-template-class-method
            Eigen::VectorXd temp(const Eigen::MatrixXd& x) const
            {
                if constexpr (std::is_same_v<Kernel, kernels::SquaredExp<Params>>) {
                    std::cout << "Hello" << std::endl;
                }

                // Check https://github.com/willwray/type_name
                std::cout << type_name_str<Kernel>() << std::endl;

                Eigen::VectorXd r(x.rows());
                return r;
            }

            /* Gradient */
            template <int Size>
            EIGEN_ALWAYS_INLINE Eigen::Matrix<double, Size, 1> grad(const Eigen::Matrix<double, Size, 1>& x) const
            {
                Eigen::Matrix<double, Size, 1> grad = Eigen::VectorXd::Zero(x.size());

                for (size_t i = 0; i < _x.rows(); i++)
                    grad += _w(i) * _k.template gradImpl<Size>(_x.row(i), x);

                return grad;
            }

            /* Gradient  multiple points */
            template <int Size>
            Eigen::Matrix<double, Eigen::Dynamic, Size> multiGrad(const Eigen::Matrix<double, Eigen::Dynamic, Size>& x) const
            {
                Eigen::Matrix<double, Eigen::Dynamic, Size> grad(x.rows(), x.cols());

#pragma omp parallel for
                for (size_t i = 0; i < grad.rows(); i++)
                    grad.row(i) = this->grad<Size>(x.row(i));

                return grad;
            }

            const Eigen::MatrixXd& samples() const { return _x; }

            const Eigen::VectorXd& weights() const { return _w; }

            Kernel& kernel() { return _k; }

            virtual Expansion& setSamples(const Eigen::MatrixXd& x)
            {
                _x = x;

                return *this;
            }

            virtual Expansion& setWeights(const Eigen::VectorXd& w)
            {
                _w = w;

                return *this;
            }

        protected:
            /* Kernel */
            Kernel _k;

            /* Parameters */
            Eigen::VectorXd _w;

            /* Referece points */
            Eigen::MatrixXd _x;
        };

        // template <>
        // Eigen::VectorXd Expansion<float, kernels::SquaredExp<float>>::temp(const Eigen::MatrixXd& x) const
        // {
        //     Eigen::VectorXd r(x.rows());

        //     return r;
        // }
    } // namespace utils
} // namespace kernel_lib

#endif // KERNELLIB_EXPANSION_UTILS_HPP