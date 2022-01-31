/*
    This file is part of kernel-lib.

    Copyright (c) 2020, 2021, 2022 Bernardo Fichera <bernardo.fichera@gmail.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/
#ifndef KERNEL_LIB_UTILS_EIGENFUNCTION_HPP
#define KERNEL_LIB_UTILS_EIGENFUNCTION_HPP

#include <Eigen/Core>
#include <unordered_map>
#include <vector>

namespace kernel_lib {
    namespace utils {
        template <typename Function>
        class EigenFunction {
        public:
            EigenFunction() {}

            // virtual inline double operator()(const Eigen::Matrix<double, size, 1>& x, const size_t& i) const = 0;

            virtual inline double operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1>& x, const double& eigenvalue) const = 0;

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

                // std::cout << _eigen_pair.size() << std::endl;
            }

        protected:
            std::vector<Function> _eigen_fun;
            std::vector<double> _eigen_val;

            std::unordered_map<double, Function> _eigen_pair;
        };

    } // namespace utils
} // namespace kernel_lib

#endif // KERNEL_LIB_UTILS_EIGENFUNCTION_HPP