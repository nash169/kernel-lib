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

#ifndef KERNELLIB_TOOLS_MATH_HPP
#define KERNELLIB_TOOLS_MATH_HPP

#include <Eigen/Dense>
#include <iostream>

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

        // Check if LAPACK works with Lower triangular as well
        Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> cholesky(const Eigen::MatrixXd& mat);

    } // namespace tools
} // namespace kernel_lib

#endif // KERNELLIB_TOOLS_MATH_HPP