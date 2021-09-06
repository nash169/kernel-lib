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

            // Epsilon neighborhoods
            Eigen::SparseMatrix<int, Eigen::RowMajor> epsNeighborhoods(const Eigen::MatrixXd& x, const double& eps = 1)
            {
                // Eigen triplet
                std::vector<Eigen::Triplet<int>> tripletList;

                for (size_t i = 0; i < x.rows(); i++)
                    for (size_t j = 0; j < x.rows(); j++) {
                        if ((x.row(i) - x.row(j)).squaredNorm() <= eps)
                            tripletList.push_back(Eigen::Triplet<int>(i, j, 1));
                    }

                return graph(tripletList, x.rows());
            }

            // Weighted Epsilon neighborhoods
            template <typename Kernel>
            Eigen::SparseMatrix<double, Eigen::RowMajor> epsNeighborhoodsWeighted(const Eigen::MatrixXd& x, const Kernel& weight, const double& eps = 1)
            {
                // Eigen triplet
                std::vector<Eigen::Triplet<double>> tripletList;

                for (size_t i = 0; i < x.rows(); i++)
                    for (size_t j = 0; j < x.rows(); j++) {
                        if ((x.row(i) - x.row(j)).squaredNorm() <= eps)
                            tripletList.push_back(Eigen::Triplet<double>(i, j, weight(x.row(i), x.row(j))));
                    }

                return graph(tripletList, x.rows());
            }

            // K-Nearest
            Eigen::SparseMatrix<int, Eigen::RowMajor> kNearest(const Eigen::MatrixXd& x, const size_t& k = 1, const size_t& exclude = 0)
            {
                // Eigen triplet
                std::vector<Eigen::Triplet<int>> tripletList;

                for (size_t i = 0; i < x.rows(); i++) {
                    std::vector<size_t> idx = sort(x.rowwise() - x.row(i));
                    for (size_t j = 0 + exclude; j < k; j++)
                        tripletList.push_back(Eigen::Triplet<int>(i, idx[j], 1));
                }

                return graph(tripletList, x.rows());
            }

            // Weighted K-Nearest
            template <typename Kernel>
            Eigen::SparseMatrix<double, Eigen::RowMajor> kNearestWeighted(const Eigen::MatrixXd& x, const Kernel& weight, const size_t& k = 1, const size_t& exclude = 0)
            {
                // Eigen triplet
                std::vector<Eigen::Triplet<double>> tripletList;

                // How to parallelize this?
                for (size_t i = 0; i < x.rows(); i++) {
                    std::vector<size_t> idx = sort(x.rowwise() - x.row(i));
                    for (size_t j = 0 + exclude; j < k; j++)
                        tripletList.push_back(Eigen::Triplet<double>(i, idx[j], weight(x.row(i), x.row(idx[j]))));
                }

                return graph(tripletList, x.rows());
            }

            // Create graph from triplet
            template <typename T>
            Eigen::SparseMatrix<T, Eigen::RowMajor> graph(const std::vector<Eigen::Triplet<T>>& tripletList, const size_t& size)
            {
                Eigen::SparseMatrix<T, Eigen::RowMajor> G = Eigen::SparseMatrix<T, Eigen::RowMajor>(size, size);
                G.setFromTriplets(tripletList.begin(), tripletList.end());

                return G;
            }

        protected:
            std::vector<size_t> sort(const Eigen::MatrixXd& v)
            {
                // initialize original index locations
                std::vector<size_t> idx(v.rows());
                std::iota(idx.begin(), idx.end(), 0);

                std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v.row(i1).squaredNorm() < v.row(i2).squaredNorm(); });

                return idx;
            }
        };

    } // namespace utils
} // namespace kernel_lib

#endif // KERNEL_LIB_UTILS_GRAPH_HPP