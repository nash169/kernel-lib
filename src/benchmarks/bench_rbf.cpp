#include <chrono>
#include <iostream>
#include <thread>

#include <kernel_lib/kernels/RiemannSqExp.hpp>
#include <kernel_lib/kernels/SquaredExp.hpp>
#include <kernel_lib/utils/Expansion.hpp>
#include <utils_cpp/UtilsCpp.hpp>

#include <type_traits>

using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, std::log(0.5));
        PARAM_SCALAR(double, sn, std::log(1.4));
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, std::log(3.1));
    };

    struct riemann_exp_sq : public defaults::riemann_exp_sq {
    };
};

// // in the .h:
// void foo(const Eigen::Ref<Eigen::MatrixXf>& A, const Eigen::Ref<Eigen::MatrixXf>& B);
// void foo(const Eigen::Ref<Eigen::MatrixXf, 0, Eigen::InnerStride<>>& A, const Eigen::Ref<Eigen::MatrixXf, 0, Eigen::InnerStride<>>& B);

// in the .cpp:
template <typename TypeOfA>
void foo_impl(const TypeOfA& a, const TypeOfA& b)
{
    std::cout << "InputType" << std::endl;
    std::cout << a.data() << std::endl;
    std::cout << b.data() << std::endl;
    // std::cout << a.dot(b) << std::endl;
}

template <int Size>
void foo(const Eigen::Ref<const Eigen::Matrix<float, Size, 1>>& a, const Eigen::Ref<const Eigen::Matrix<float, Size, 1>>& b)
{
    std::cout << "no stride" << std::endl;
    foo_impl(a, b);
}

template <int Size>
void foo(const Eigen::Ref<const Eigen::Matrix<float, Size, 1>, 0, Eigen::InnerStride<>>& a, const Eigen::Ref<const Eigen::Matrix<float, Size, 1>, 0, Eigen::InnerStride<>>& b)
{
    std::cout << "stride" << std::endl;
    foo_impl(a, b);
}

int main(int argc, char const* argv[])
{
    constexpr int dim = 2, num_samples = 20000;

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, dim), Y = Eigen::MatrixXd::Random(num_samples, dim);
    // Eigen::VectorXd a = Eigen::VectorXd::Random(dim), b = Eigen::VectorXd::Random(dim);

    // Eigen::Matrix<double, Eigen::Dynamic, dim> X = Eigen::MatrixXd::Random(num_samples, dim), Y = Eigen::MatrixXd::Random(num_samples, dim);
    // Eigen::Matrix<double, dim, 1> a = Eigen::VectorXd::Random(dim), b = Eigen::VectorXd::Random(dim);
    Eigen::VectorXd W = Eigen::VectorXd::Random(num_samples), p = Eigen::VectorXd::Random(dim);

    using Kernel_t = kernels::SquaredExp<Params>;
    Kernel_t k;

    // std::cout << "BENCHMARK: Kernel evaluation" << std::endl;
    // {
    //     utils_cpp::Timer timer;
    //     k.gram<dim>(X, Y);
    // }

    using Expansion_t = utils::Expansion<Params, Kernel_t>;
    Expansion_t psi;

    psi.temp(X);

    // psi.setReference(X).setParams(W);
    // {
    //     utils_cpp::Timer timer;
    //     psi.multiEval(Y);
    // }

    // using Riemann_t = kernels::RiemannSqExp<Params, Expansion_t>;
    // Riemann_t riemann;

    // for (size_t i = 0; i < 100; i++) {
    //     Expansion_t f;
    //     f.setReference(Eigen::MatrixXd::Random(20000, 2)).setParams(Eigen::VectorXd::Random(20000));
    //     double a = 1.2;
    //     riemann.addEigenPair(a, f);
    // }

    // {
    //     utils_cpp::Timer timer;
    //     riemann.temp(X, Y);
    // }

    // std::cout << sizeof(float) << std::endl;
    // std::cout << std::alignment_of<int>() << std::endl;

    // Eigen::MatrixXf A = Eigen::MatrixXf::Random(6, dim);
    // Eigen::VectorXf a = Eigen::VectorXf::Random(dim);

    // utils_cpp::FileManager io_manager;

    // io_manager.setFile("rsc/test_point.csv").write("x", a, "X", A);

    // Eigen::MatrixXd X = io_manager.setFile("rsc/test_point.csv").read<Eigen::MatrixXd>("X", 2);
    // Eigen::VectorXd x = io_manager.setFile("rsc/test_point.csv").read<Eigen::MatrixXd>("x", 2);

    // std::cout << A.data() << std::endl;
    // std::cout << A.row(0).data() << std::endl;

    // foo<dim>(A.row(1), A.row(2));

    // k.temp3(x, x);
    // k.temp4(x, x);
    // k.gram(X, X);
    // constexpr int Size = -1;
    // Eigen::VectorXd test = (Eigen::Matrix<double, Size, 1>() << 1, 2, 3).finished();

    // std::cout << test.transpose() << std::endl;

    return 0;
}