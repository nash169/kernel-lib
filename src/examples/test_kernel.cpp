#include <kernel_lib/rbf.hpp>
#include <kernel_lib/tools/timer.hpp>

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
    int dim = 2, num_points = 15000;

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_points, dim);

    Rbf<Params> k;

    {
        tools::Timer timer;
        k(x, x);
    }

    return 0;
}
