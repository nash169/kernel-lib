#include <Eigen/Dense>
#include <iostream>
#include <kernel_lib/kernel.hpp>
#include <kernel_lib/tools/macros.hpp>
#include <kernel_lib/tools/math.hpp>

#include <chrono>
#include <cmath>
#include <random>

using namespace kernel_lib;

struct model_gmm {
    /// @ingroup model_gmm_defaults
    BO_PARAM(double, full_covariances, 1.4);
};

struct Params {
    struct kernel : public defaults::kernel {
    };
};

int main(int argc, char const* argv[])
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    // generate N random numbers
    int N = 1000;

    // the incorrect way
    FILE* incorrect;
    incorrect = fopen("incorrect.csv", "w");
    fprintf(incorrect, "Theta,Phi,x,y,z\n");
    for (int i = 0; i < N; i++) {
        // incorrect way
        double theta = 2 * M_PI * uniform01(generator);
        double phi = M_PI * uniform01(generator);
        double x = sin(phi) * cos(theta);
        double y = sin(phi) * sin(theta);
        double z = cos(phi);
        fprintf(incorrect, "%f,%f,%f,%f,%f\n", theta, phi, x, y, z);
    }
    fclose(incorrect);

    bool optim_noise = true;
    double var = 0.3;

    Kernel<Params> test_ker(optim_noise, var);

    // std::cout << test_ker.noise() << std::endl;
    // std::cout << test_ker.var() << std::endl;

    Eigen::VectorXd v(6);
    v << 1, 2, 3, 4, 5, 6;

    Eigen::Map<Eigen::MatrixXd> M(v.data(), 3, 2);
    // std::cout << M << std::endl;

    Eigen::MatrixXd T = tools::c_reshape<Eigen::VectorXd, Eigen::MatrixXd>(v, 3, 2);
    std::cout << T << std::endl;

    return 0;
}
