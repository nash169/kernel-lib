#ifndef KERNEL_LIB_UTILS_GRAPH_HPP
#define KERNEL_LIB_UTILS_GRAPH_HPP

#include <Eigen/Sparse>
#include <algorithm> // std::sort, std::stable_sort
#include <numeric> // std::iota
#include <vector>

namespace kernel_lib {
    namespace utils {
        class Graph {
        public:
            Graph() {}

            const Eigen::SparseMatrix<int, Eigen::RowMajor>& graph() const
            {
                return _G;
            }

            Graph& epsNeighborhoods(const Eigen::MatrixXd& x, const double& eps = 1)
            {
                // Eigen triplet
                std::vector<Eigen::Triplet<int>> tripletList;

                for (size_t i = 0; i < x.rows(); i++)
                    for (size_t j = 0; j < x.rows(); j++) {
                        if ((x.row(i) - x.row(j)).squaredNorm() <= eps)
                            tripletList.push_back(Eigen::Triplet<int>(i, j, 1));
                    }

                _G = Eigen::SparseMatrix<int, Eigen::RowMajor>(x.rows(), x.rows());
                _G.setFromTriplets(tripletList.begin(), tripletList.end());

                return *this;
            }

            Graph& kNearest(const Eigen::MatrixXd& x, const size_t& k = 1)
            {
                // Eigen triplet
                std::vector<Eigen::Triplet<int>> tripletList;

                for (size_t i = 0; i < x.rows(); i++) {
                    std::vector<size_t> idx = sort(x.rowwise() - x.row(i));
                    for (size_t j = 0; j < k; j++)
                        tripletList.push_back(Eigen::Triplet<int>(i, idx[j], 1));
                }

                _G = Eigen::SparseMatrix<int, Eigen::RowMajor>(x.rows(), x.rows());
                _G.setFromTriplets(tripletList.begin(), tripletList.end());

                return *this;
            }

            std::vector<size_t> sort(const Eigen::MatrixXd& v)
            {
                // initialize original index locations
                std::vector<size_t> idx(v.rows());
                std::iota(idx.begin(), idx.end(), 0);

                std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v.row(i1).squaredNorm() < v.row(i2).squaredNorm(); });

                return idx;
            }

        protected:
            Eigen::SparseMatrix<int, Eigen::RowMajor> _G;
        };

    } // namespace utils
} // namespace kernel_lib

#endif // KERNEL_LIB_UTILS_GRAPH_HPP