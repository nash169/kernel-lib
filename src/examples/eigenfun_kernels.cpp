#include <iostream>
#include <utils_cpp/UtilsCpp.hpp>

#include <kernel_lib/kernels/RiemannSqExp.hpp>
#include <kernel_lib/utils/EigenFunction.hpp>

using namespace kernel_lib;

// Kernel parameters
struct Params {
    struct kernel : public defaults::kernel {
    };

    struct riemann_exp_sq : public defaults::riemann_exp_sq {
    };
};

// Example of eigen-function
class SinFunction {
public:
    SinFunction(double h = 1.0) : _h(h) {}

    double eval(const Eigen::VectorXd& x) const
    {
        return std::sin(_h * x.squaredNorm());
    }

protected:
    double _h;
};

// Set of eigen-functions to pass to the kernel
template <int size>
class MyFunction : public utils::EigenFunction<size, SinFunction> {
public:
    MyFunction()
    {
        // // Add 3 eigen-functions of type SinFunction
        // utils::EigenFunction<size, SinFunction>::addEigenFunctions(SinFunction(0.3), SinFunction(1.2), SinFunction(0.7));

        // // Add 3 random eigen-values
        // utils::EigenFunction<size, SinFunction>::addEigenValues(0.5, 0.2, 0.8);

        // Add unordered map
        utils::EigenFunction<size, SinFunction>::addEigenPair(0.5, SinFunction(0.3), 0.2, SinFunction(1.2), 0.8, SinFunction(0.7));
    }

    // // Override operator()
    // inline double operator()(const Eigen::Matrix<double, size, 1>& x, const size_t& i) const override
    // {
    //     return utils::EigenFunction<size, SinFunction>::_f[i].eval(x);
    // }

    // Override operator()
    inline double operator()(const Eigen::Matrix<double, size, 1>& x, const double& eigenvalue) const override
    {
        return utils::EigenFunction<size, SinFunction>::_eigen_pair.at(eigenvalue).eval(x);
    }

protected:
};

int main(int argc, char const* argv[])
{
    double box[] = {0, 100, 0, 100};
    size_t resolution = 100, num_samples = resolution * resolution, dim = sizeof(box) / sizeof(*box) / 2;

    Eigen::MatrixXd X(resolution, resolution), Y(resolution, resolution), x_test(num_samples, dim), x_train(1, dim);

    // Test set
    X = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1);
    Y = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution);
    x_test << Eigen::Map<Eigen::VectorXd>(X.data(), X.size()), Eigen::Map<Eigen::VectorXd>(Y.data(), Y.size());

    using Kernel_t = kernels::RiemannSqExp<Params, MyFunction<2>>;
    Kernel_t k;

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(6, 2);

    std::cout << k(x, x) << std::endl;
    return 0;
}
