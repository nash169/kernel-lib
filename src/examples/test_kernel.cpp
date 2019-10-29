#include <Eigen/Dense>
#include <iostream>
#include <kernel_lib/rbf.hpp>
#include <kernel_lib/tools/math.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
    };
    struct kernel_rbf : public defaults::kernel_rbf {
    };
};

int main(int argc, char const* argv[])
{
    Rbf<Params> myrbf;

    Eigen::VectorXd v(6);
    v << 1, 2, 3, 4, 5, 6;

    Eigen::Map<Eigen::MatrixXd> M(v.data(), 3, 2);
    // std::cout << M << std::endl;

    Eigen::MatrixXd T = tools::c_reshape<Eigen::VectorXd, Eigen::MatrixXd>(v, 3, 2);
    std::cout << T << std::endl;

    return 0;
}
