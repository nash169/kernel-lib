#ifndef KERNELLIB_EXPANSION_UTILS_HPP
#define KERNELLIB_EXPANSION_UTILS_HPP

#include "kernel_lib/kernels/Exp.hpp"

namespace kernel_lib {
    namespace defaults {
        struct expansion {
            PARAM_SCALAR(bool, reference_first, true)
        };
    } // namespace defaults

    namespace utils {
        template <typename Params, typename Kernel = kernels::Exp<Params>>
        class Expansion {
        public:
            Expansion() : _kernel(), _reference_first(Params::expansion::reference_first()) {}

            Eigen::VectorXd operator()(const Eigen::MatrixXd& x)
            {
                if (!_weight.size())
                    _weight = Eigen::VectorXd::Ones(_reference.rows());
                else
                    REQUIRED_DIMENSION(_reference.rows() == _weight.rows(), "The number of weights must match the number of reference points (first input)")

                Eigen::MatrixXd psi;

                if (_reference_first)
                    psi = _kernel(_reference, x).reshaped(_reference.rows(), x.rows()).transpose();
                else
                    psi = _kernel(x, _reference).reshaped(x.rows(), _reference.rows()).transpose();

                for (size_t i = 0; i < psi.rows(); i++)
                    psi.row(i) = psi.row(i).cwiseProduct(_weight.transpose());

                return psi.rowwise().sum();
            }

            Eigen::VectorXd operator()(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
            {
                if (_reference_first) {
                    setReference(x, true);
                    return (*this)(y);
                }
                else {
                    setReference(y, false);
                    return (*this)(x);
                }
            }

            void setReference(const Eigen::MatrixXd& reference, bool reference_first = true)
            {
                _reference = reference;
                _reference_first = reference_first;
            }

            void setWeights(const Eigen::VectorXd& weights)
            {
                _weight = weights;
            }

            Kernel& kernel()
            {
                return _kernel;
            }

        protected:
            Kernel _kernel;

            Eigen::VectorXd _weight;
            Eigen::MatrixXd _reference;

            bool _reference_first;
        };
    } // namespace utils
} // namespace kernel_lib

#endif // KERNELLIB_EXPANSION_UTILS_HPP