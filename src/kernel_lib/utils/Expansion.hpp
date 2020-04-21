#ifndef KERNELLIB_EXPANSION_UTILS_HPP
#define KERNELLIB_EXPANSION_UTILS_HPP

#include "kernel_lib/kernels/Exp.hpp"

namespace kernel_lib {
    namespace defaults {
        struct expansion {
            PARAM_SCALAR(bool, reference, true)
            PARAM_VECTOR(double, weight, 5)
        };
    } // namespace defaults

    namespace utils {
        template <typename Params, typename Kernel = kernels::Exp<Params>>
        class Expansion {
        public:
            Expansion() : _kernel(), _weight(Params::expansion::weight()), _reference(Params::expansion::reference())
            {
            }

            Eigen::VectorXd operator()(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                if (!_reference)
                    REQUIRED_DIMENSION(x.rows() == _weight.rows(), "The number of weights must match the number of reference points (first input)")
                else
                    REQUIRED_DIMENSION(y.rows() == _weight.rows(), "The number of weights must match the number of reference points (second input)")

                Eigen::MatrixXd psi(y.rows(), x.rows());
                psi << _kernel(x, y).reshaped(y.rows(), x.rows());

                for (size_t i = 0; i < psi.rows(); i++)
                    psi.row(i) = psi.row(i).cwiseProduct(_weight.transpose());

                return psi.rowwise().sum();
            }

        protected:
            Kernel _kernel;

            Eigen::VectorXd _weight;

            bool _reference;
        };
    } // namespace utils
} // namespace kernel_lib

#endif // KERNELLIB_EXPANSION_UTILS_HPP