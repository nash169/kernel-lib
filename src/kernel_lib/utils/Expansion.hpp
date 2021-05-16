#ifndef KERNELLIB_EXPANSION_UTILS_HPP
#define KERNELLIB_EXPANSION_UTILS_HPP

namespace kernel_lib {
    namespace defaults {
        struct expansion {
        };
    } // namespace defaults

    namespace utils {
        template <typename Params, typename Kernel>
        class Expansion {
        public:
            Expansion() : _kernel() {}

            double operator()(const Eigen::VectorXd& x)
            {
                double r = 0;

                for (size_t i = 0; i < _reference.rows(); i++)
                    r += _weights(i) * _kernel(_reference(i), x);

                return r;
            }

            Eigen::VectorXd multiEval(const Eigen::MatrixXd& x)
            {
                Eigen::VectorXd r(x.rows());

                for (size_t i = 0; i < r.rows(); i++)
                    r(i) = (*this)(x.row(i));

                return r;
            }

            Expansion& setReference(const Eigen::MatrixXd& reference)
            {
                _reference = reference;

                return *this;
            }

            Expansion& setWeights(const Eigen::VectorXd& weights)
            {
                _weights = weights;

                return *this;
            }

            Kernel& kernel()
            {
                return _kernel;
            }

        protected:
            Kernel _kernel;

            Eigen::VectorXd _weights;
            Eigen::MatrixXd _reference;
        };
    } // namespace utils
} // namespace kernel_lib

#endif // KERNELLIB_EXPANSION_UTILS_HPP