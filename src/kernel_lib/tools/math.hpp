#ifndef KERNELLIB_TOOLS_MATH_HPP
#define KERNELLIB_TOOLS_MATH_HPP

#include <Eigen/Dense>
#include <iostream>

using Chol = Eigen::LLT<Eigen::MatrixXd>;

namespace kernel_lib {
    namespace tools {
        Eigen::MatrixXd c_reshape(Eigen::MatrixXd M, int num_rows, int num_cols);

        Eigen::MatrixXd repeat(Eigen::MatrixXd M, int num_rows, int num_cols);

        Eigen::MatrixXd matrix_transpose(Eigen::MatrixXd& M);

        Eigen::MatrixXd matrix_product(Eigen::MatrixXd& A, Eigen::MatrixXd& B);

        Eigen::MatrixXd outer_product(Eigen::MatrixXd& A, Eigen::MatrixXd& B);

        Eigen::MatrixXd blkdiag_matrix(Eigen::MatrixXd& M);

        Eigen::MatrixXd blkdiag_revert(Eigen::MatrixXd& M, int dim);

        Eigen::MatrixXd gramSchmidt(const Eigen::MatrixXd& V);

        Eigen::MatrixXd createCovariance(const Eigen::VectorXd& direction, const Eigen::VectorXd& std, bool inverse = false);

        Eigen::VectorXd linearMap(Eigen::VectorXd x, double xmin, double xmax, double ymin, double ymax);

        Eigen::MatrixXd repeat_block(Eigen::MatrixXd& M, int blksize, int repeat, int direction);

        // Code from limbo to calculate the cholesky (check if optimized)
        // also check if it is ok returning inside the is statement
        // with this thing compilation time gets super long why?
        Chol::Traits::MatrixL cholesky(const Eigen::MatrixXd& sigma);

        Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> upperCholesky(const Eigen::MatrixXd& mat);
    } // namespace tools
} // namespace kernel_lib

#endif // KERNELLIB_TOOLS_MATH_HPP