#ifndef KERNEL_LIB_UTILS_EIGENFUNCTION_HPP
#define KERNEL_LIB_UTILS_EIGENFUNCTION_HPP

#include <Eigen/Core>
#include <unordered_map>

namespace kernel_lib {
    namespace utils {
        template <int size, typename Function>
        class EigenFunction {
        public:
            EigenFunction() {}

            // virtual inline double operator()(const Eigen::Matrix<double, size, 1>& x, const size_t& i) const = 0;

            virtual inline double operator()(const Eigen::Matrix<double, size, 1>& x, const double& eigenvalue) const = 0;

            std::vector<Function>& eigenFunctions()
            {
                return _eigen_fun;
            }

            std::vector<double>& eigenValues()
            {
                return _eigen_val;
            }

            const std::unordered_map<double, Function>& eigenPair() const
            {
                return _eigen_pair;
            }

            // First way of setting variadic templates in vectors
            template <typename T, typename... Ts>
            void addEigenFunctions(T value, Ts... args)
            {
                _eigen_fun.push_back(value);
                if constexpr (sizeof...(args) > 0)
                    addEigenFunctions(args...);
            }

            // second way of setting variadic templates in vectors
            template <typename... Args>
            void addEigenValues(Args... args)
            {
                _eigen_val = {args...};
            }

            template <typename... Args>
            void addEigenPair(const double& eigenvalue, const Function& eigenfunction, Args... args)
            {
                _eigen_pair.insert(std::make_pair(eigenvalue, eigenfunction));
                if constexpr (sizeof...(args) > 0)
                    addEigenPair(args...);
            }

        protected:
            std::vector<Function> _eigen_fun;
            std::vector<double> _eigen_val;

            std::unordered_map<double, Function> _eigen_pair;
        };

    } // namespace utils
} // namespace kernel_lib

#endif // KERNEL_LIB_UTILS_EIGENFUNCTION_HPP