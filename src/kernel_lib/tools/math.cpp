#include "kernel_lib/tools/math.hpp"
#include <Eigen/Dense> // use Eigen/Dense for now, then it'd be better to specialize headers

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

        Eigen::MatrixXd repeat_block(Eigen::MatrixXd& M, int blksize, int repeat, int direction)
        {
            return M;
        }

        Eigen::MatrixXd gramSchmidt(const Eigen::MatrixXd& V)
        {
            size_t n_points = V.rows(), n_features = V.cols();

            Eigen::MatrixXd M(n_points * n_features, n_features);

            for (size_t i = 0; i < n_points; i++) {
                Eigen::MatrixXd mat(n_features, n_features);

                for (size_t j = 0; j < n_features; j++)
                    mat.col(j) = V.row(i).transpose();

                Eigen::HouseholderQR<Eigen::MatrixXd> qr(mat);

                M.block(i * n_features, 0, n_features, n_features) = qr.householderQ();
            }

            return -M;
        }

        Eigen::VectorXd linearMap(Eigen::VectorXd x, double xmin, double xmax, double ymin, double ymax)
        {
            double m = (ymin - ymax) / (xmin - xmax), q = ymin - m * xmin;

            Eigen::VectorXd y(x.rows());

            for (size_t i = 0; i < y.rows(); i++)
                y(i) = m * x(i) + q;

            return y;
        }

        Eigen::MatrixXd createCovariance(const Eigen::VectorXd& direction, const Eigen::VectorXd& std, bool inverse)
        {
            size_t dim = direction.rows();

            Eigen::MatrixXd U = gramSchmidt(direction.transpose()),
                            D = Eigen::MatrixXd::Identity(dim, dim);

            if (inverse)
                D.diagonal() = std.array().pow(2).inverse();
            else
                D.diagonal() = std.array().pow(2);

            return U * D * U.transpose(); // U.transpose() * D * U; for the inverse ?
        }

        Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> cholesky(const Eigen::MatrixXd& mat)
        {
            Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> uut = mat.selfadjointView<Eigen::Upper>().llt();

            if (uut.info() != Eigen::Success) {
                // There was an error; probably the matrix is not SPD
                // Let's try to make it SPD and take cholesky of that
                // original MATLAB code: http://fr.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
                // Note that at this point _L is not cholesky factor, but matrix to be factored

                // Symmetrize A into B
                Eigen::MatrixXd B = (mat.array() + mat.transpose().array()) / 2.;

                // Compute the symmetric polar factor of B. Call it H. Clearly H is itself SPD.
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::MatrixXd V, Sigma, H, U_hat;

                Sigma = Eigen::MatrixXd::Identity(B.rows(), B.cols());
                Sigma.diagonal() = svd.singularValues();
                V = svd.matrixV();

                H = V * Sigma * V.transpose();

                // Get candidate for closest SPD matrix to sigma
                U_hat = (B.array() + H.array()) / 2.;

                // Ensure symmetry
                U_hat = (U_hat.array() + U_hat.array()) / 2.;

                // Test that U_hat is in fact PD. if it is not so, then tweak it just a bit.
                Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> uut_hat = U_hat.selfadjointView<Eigen::Upper>().llt();

                int k = 0;
                while (uut_hat.info() != Eigen::Success) {
                    k++;
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(U_hat);
                    double min_eig = es.eigenvalues().minCoeff();
                    U_hat.diagonal().array() += (-min_eig * k * k + 1e-50);
                    uut_hat.compute(U_hat);
                }

                return uut_hat;
            }
            else
                return uut;
        }
    } // namespace tools
} // namespace kernel_lib