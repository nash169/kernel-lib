#ifndef KERNELLIB_TOOLS_HELPER_HPP
#define KERNELLIB_TOOLS_HELPER_HPP

#include <Eigen/Core>

namespace kernel_lib {
    namespace tools {
        Eigen::VectorXd makeVector(double num)
        {
            Eigen::VectorXd vec(1);
            vec << num;

            return vec;
        }
    } // namespace tools
} // namespace kernel_lib

#endif // KERNELLIB_TOOLS_HELPER_HPP