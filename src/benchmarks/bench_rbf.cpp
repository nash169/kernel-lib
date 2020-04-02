#include <kernel_lib/Expansion.hpp>
#include <kernel_lib/kernel/Rbf.hpp>

#include <kernel_lib/tools/Timer.hpp>
#include <kernel_lib/tools/math.hpp>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sigma_n, 1.0);
    };
    struct kernel_rbf : public defaults::kernel_rbf {
        PARAM_SCALAR(Covariance, type, CovarianceType::FULL);
        PARAM_SCALAR(bool, inverse, false);
        PARAM_VECTOR(double, sigma, 14.5, -10.5, -10.5, 14.5); // 14.5, -10.5, -10.5, 14.5 -- 0.145, 0.105, 0.105, 0.145
    };
    struct expansion : public defaults::expansion {
        PARAM_VECTOR(double, weight, 1);
    };
};

int main(int argc, char const* argv[])
{
    size_t dim = 2, num_samples = 10000;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, dim);

    using Kernel_t = kernel::Rbf<Params>;

    Kernel_t k;
    {
        tools::Timer timer;
        k(X, X);
    }

    return 0;
}