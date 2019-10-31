#ifndef KERNEL_LIB_TOOLS_MATH_HPP
#define KERNEL_LIB_TOOLS_MATH_HPP

#include <Eigen/Core>
#include <iostream>

namespace kernel_lib {
    namespace tools {
        Eigen::MatrixXd c_reshape(Eigen::MatrixXd M, int num_rows, int num_cols)
        {
            M.transposeInPlace();

            Eigen::Map<Eigen::MatrixXd> S(M.data(), num_cols, num_rows);

            return S.transpose();
        }

        Eigen::MatrixXd repeat(Eigen::MatrixXd M, int num_rows, int num_cols)
        {
            // Rows repeat
            Eigen::MatrixXd T = c_reshape(M.replicate(1, num_rows), M.rows() * num_rows, M.cols());

            Eigen::Map<Eigen::MatrixXd> S(Eigen::MatrixXd(T.replicate(num_cols, 1)).data(), T.rows(), T.cols() * num_cols);

            return S;
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