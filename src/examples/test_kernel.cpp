#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <kernel_lib/rbf.hpp>
#include <kernel_lib/tools/math.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        BO_PARAM(double, sigma_n, 1.0);
    };
    struct kernel_rbf : public defaults::kernel_rbf {
    };
};

int main(int argc, char const* argv[])
{
    int dim = 2, num_points = 20000;

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_points, dim);

    // x << 0.8147, 0.9134,
    //     0.9058, 0.6324,
    //     0.1270, 0.0975;

    Rbf<Params> k;
    // k(x, x);

    auto t1 = std::chrono::high_resolution_clock::now();
    k(x, x);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::cout << duration << std::endl;

    // Eigen::MatrixXd v(6, 1);
    // v << 1, 2, 3, 4, 5, 6;

    // Eigen::MatrixXd T = tools::c_reshape(v, 3, 2);

    // std::cout << T.rowwise().sum() << std::endl;

    // Eigen::MatrixXd M = tools::repeat(T, 2, 2);

    // std::cout << M << std::endl;

    // std::cout << T << std::endl;

    // T.transposeInPlace();

    // Eigen::Map<Eigen::MatrixXd> S(T.data(), 2, 6);

    // Eigen::Map<Eigen::MatrixXd> S(Eigen::MatrixXd(T.transpose()).data(), 2, 6);

    // std::cout << tools::c_reshape<Eigen::MatrixXd, Eigen::MatrixXd>(T, 6, 2)
    //           << std::endl;

    // auto T2 = tools::c_reshape<Eigen::MatrixXd, Eigen::MatrixXd>(T, 6, 2);

    // std::cout << T2 << std::endl;

    // Eigen::MatrixXd test = Eigen::MatrixXd::Random(4, 2);

    // Eigen::VectorXd test2 = test.rowwise().norm();

    // Eigen::Matrix<bool, Eigen::Dynamic, 1> var = test2.array() >= 0.7;

    // std::cout << var.cast<double>() * 5.0 << std::endl;

    return 0;
}
