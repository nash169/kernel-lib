#include <iostream>
#include <kernel_lib/Kernel.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel_exp : public defaults::kernel_exp {
        PARAM_SCALAR(Covariance, type, CovarianceType::SPHERICAL); // CovarianceType::FULL
        PARAM_SCALAR(bool, inverse, false);
        PARAM_VECTOR(double, sigma, 1); // 14.5, -10.5, -10.5, 14.5 -- 0.145, 0.105, 0.105, 0.145
    };
};

int main(int argc, char const* argv[])
{
    size_t dim = 2, num_samples = 20000;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, dim);

    using Kernel_t = kernels::Exp<Params>;

    Kernel_t k;
    {
        utils::Timer timer;
        k(X, X);
    }

    return 0;
}