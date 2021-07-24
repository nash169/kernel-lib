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
            Expansion() : _kernel() {}

            template <int Size>
            EIGEN_ALWAYS_INLINE double operator()(const Eigen::Matrix<double, Size, 1>& x) const
            {
                double r = 0;

                // Can parallelize this?
                // https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-data.html
                for (size_t i = 0; i < _reference.rows(); i++)
                    r += _params(i) * _kernel.template kernelImpl<Size>(_reference.row(i), x);

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
                return _kernel.gram(_reference, x).transpose() * _params;
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

            const Eigen::MatrixXd& reference() const
            {
                return _reference;
            }

            const Eigen::VectorXd& params() const
            {
                return _params;
            }

            virtual Expansion& setReference(const Eigen::MatrixXd& reference)
            {
                _reference = reference;

                return *this;
            }

            virtual Expansion& setParams(const Eigen::VectorXd& params)
            {
                _params = params;

                return *this;
            }

            Kernel& kernel()
            {
                return _kernel;
            }

        protected:
            /* Kernel */
            Kernel _kernel;

            /* Parameters */
            Eigen::VectorXd _params;

            /* Referece points */
            Eigen::MatrixXd _reference;
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