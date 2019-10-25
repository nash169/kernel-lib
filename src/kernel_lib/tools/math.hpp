#ifndef KERNEL_LIB_TOOLS_MATH_HPP
#define KERNEL_LIB_TOOLS_MATH_HPP

#include <Eigen/Core>

namespace kernel_lib {
    namespace tools {
        template <typename Input, typename Output>
        Output c_reshape(Input& M, int num_rows, int num_cols)
        {
            Eigen::Map<Output> S(M.transpose().data(), num_cols, num_rows);

            return S.transpose();
        }

        Eigen::MatrixXd matrix_transpose(Eigen::MatrixXd& M)
        {
            return M;
        }

        Eigen::MatrixXd matrix_product(Eigen::MatrixXd& A, Eigen::MatrixXd& B)
        {
            return A;
        }

        Eigen::MatrixXd outer_product(Eigen::MatrixXd& A, Eigen::MatrixXd& B)
        {
            return A;
        }

        Eigen::MatrixXd blkdiag_matrix(Eigen::MatrixXd& M)
        {
            return M;
        }

        Eigen::MatrixXd blkdiag_revert(Eigen::MatrixXd& M, int dim)
        {
            return M;
        }

        Eigen::MatrixXd gs_orthogonalize(Eigen::MatrixXd& M)
        {
            return M;
        }

        Eigen::MatrixXd repeat_block(Eigen::MatrixXd& M, int blksize, int repeat, int direction)
        {
            return M;
        }
    } // namespace tools
} // namespace kernel_lib

#endif // KERNEL_LIB_TOOLS_MATH_HPP