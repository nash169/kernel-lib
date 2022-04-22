/*
    This file is part of kernel-lib.

    Copyright (c) 2020, 2021, 2022 Bernardo Fichera <bernardo.fichera@gmail.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <iostream>

#include <kernel_lib/Kernel.hpp>

#include <utils_lib/DerivativeChecker.hpp>
#include <utils_lib/FileManager.hpp>

using namespace utils_lib;
using namespace kernel_lib;

// Kernel
#define EXPANSION utils::Expansion<ParamsEigenfunction, kernels::SquaredExp<ParamsEigenfunction>>

#define RIEMANNSQUAREDEXP kernels::RiemannSqExp<ParamsKernel, EXPANSION>
#define RIEMANNMATERN kernels::RiemannMatern<ParamsKernel, EXPANSION>

#define KERNEL RIEMANNSQUAREDEXP

struct ParamsKernel {

    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.5);
        PARAM_SCALAR(double, sn, 1.4);
    };

    struct riemann_exp_sq : public defaults::riemann_exp_sq {
        PARAM_SCALAR(double, l, 1);
    };
};

struct ParamsEigenfunction {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0);
        PARAM_SCALAR(double, sn, -5);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, -2.30259);
    };
};

/* TOTAL KERNEL: Function in X */
template <int size>
struct FunctionX {
    FunctionX(KERNEL& k) : _k(k)
    {
        _x.setOnes();
        _x.normalized();
    }

    virtual double operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return _k(x, _x);
    }

    Eigen::Matrix<double, size, 1> _x;
    KERNEL& _k;
};

/* TOTAL KERNEL: Gradient in X */
template <int size>
struct GradientX : public KERNEL {
    GradientX(KERNEL& k) : _k(k)
    {
        _x.setOnes();
        _x.normalized();
    }

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return _k.grad(x, _x);
    }

    Eigen::Matrix<double, size, 1> _x;
    KERNEL& _k;
};

/* TOTAL KERNEL: Function in Y */
template <int size>
struct FunctionY : public KERNEL {
    FunctionY(KERNEL& k) : _k(k)
    {
        _x.setOnes();
        _x.normalized();
    }

    virtual double operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return _k(_x, x);
    }

    Eigen::Matrix<double, size, 1> _x;
    KERNEL& _k;
};

/* TOTAL KERNEL: Gradient in Y */
template <int size>
struct GradientY : public KERNEL {
    GradientY(KERNEL& k) : _k(k)
    {
        _x.setOnes();
        _x.normalized();
    }

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return _k.grad(_x, x);
    }

    Eigen::Matrix<double, size, 1> _x;
    KERNEL& _k;
};

/* TOTAL KERNEL: Function in PARAMS */
template <int size>
struct FunctionParamsT : public KERNEL {
    FunctionParamsT(KERNEL& k) : _k(k)
    {
        _x.setOnes();
        _x.normalized();
        _y.setConstant(-1);
        _y.normalized();
    }

    double operator()(const Eigen::VectorXd& params)
    {
        _k.setParams(params);

        return _k(x, y);
    }

    Eigen::Matrix<double, size, 1> _x, _y;
    KERNEL& _k;
};

/* TOTAL KERNEL: Gradient in PARAMS */
template <int size>
struct GradientParamsT : public KERNEL {
    GradientParamsT(KERNEL& k) : _k(k)
    {
        _x.setOnes();
        _x.normalized();
        _y.setConstant(-1);
        _y.normalized();
    }

    Eigen::VectorXd operator()(const Eigen::VectorXd& params)
    {
        _k.setParams(params);

        return _k.gradParams(x, y);
    }

    Eigen::Matrix<double, size, 1> _x, _y;
    KERNEL& _k;
};

/* SPECIFIC KERNEL: Function in PARAMS */
template <int size>
struct FunctionParamsS : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    double operator()(const Eigen::VectorXd& params)
    {
        KERNEL::setParameters(params);

        return KERNEL::kernel(x, y);
    }
};

/* SPECIFIC KERNEL: Gradient in PARAMS */
template <int size>
struct GradientParamsS : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    Eigen::VectorXd operator()(const Eigen::VectorXd& params)
    {
        KERNEL::setParameters(params);

        if constexpr (std::is_same_v<decltype(KERNEL::gradientParams(x, y)), double>)
            return tools::makeVector(KERNEL::gradientParams(x, y));
        else
            return KERNEL::gradientParams(x, y);
    }
};

int main(int argc, char const* argv[])
{
    using Kernel_t = KERNEL;
    Kernel_t k;
    constexpr int dim = 3;

    // Samples on manifold
    FileManager mn;
    Eigen::MatrixXd X = mn.setFile("rsc/nodes.csv").read<Eigen::MatrixXd>();
    Eigen::VectorXd D = mn.setFile("rsc/eigval.csv").read<Eigen::MatrixXd>();
    Eigen::MatrixXd U = mn.setFile("rsc/eigvec.csv").read<Eigen::MatrixXd>().transpose();
    for (size_t i = 0; i < 10; i++) {
        EXPANSION f; // Create eigenfunction
        f.setSamples(X).setWeights(U.col(i)); // Set manifold sampled points and weights
        k.addPair(D(i), f); // Add eigen-pair to Riemann kernel
    }

    Eigen::VectorXd x = Eigen::VectorXd::Random(dim);
    x.normalized();

    FunctionX<dim> f(k);

    std::cout << "TOTAL KERNEL: Function in X test" << std::endl;
    std::cout << f(x) << std::endl;

    // std::cout << "TOTAL KERNEL: Gradient in X test" << std::endl;
    // std::cout << GradientX<dim>()(x).transpose() << std::endl;

    // std::cout << "TOTAL KERNEL: Function in Y test" << std::endl;
    // std::cout << FunctionY<dim>()(x) << std::endl;

    // std::cout << "TOTAL KERNEL: Gradient in Y test" << std::endl;
    // std::cout << GradientY<dim>()(x).transpose() << std::endl;

    // std::cout << "TOTAL KERNEL: Function in PARAMS" << std::endl;
    // std::cout << FunctionParamsT<dim>()(params_t) << std::endl;

    // std::cout << "TOTAL KERNEL:: Gradient in PARAMS" << std::endl;
    // std::cout << GradientParamsT<dim>()(params_t).transpose() << std::endl;

    // std::cout << "SPECIFIC KERNEL: Function in PARAMS" << std::endl;
    // std::cout << FunctionParamsS<dim>()(params_s) << std::endl;

    // std::cout << "SPECIFIC KERNEL: Gradient in PARAMS" << std::endl;
    // std::cout << GradientParamsS<dim>()(params_s).transpose() << std::endl;

    // DerivativeChecker checker(dim);

    // if (checker.checkGradient(FunctionX<dim>(), GradientX<dim>()))
    //     std::cout << "The X gradient is CORRECT!" << std::endl;
    // else
    //     std::cout << "The X gradient is NOT correct!" << std::endl;

    // if (checker.checkGradient(FunctionY<dim>(), GradientY<dim>()))
    //     std::cout << "The Y gradient is CORRECT!" << std::endl;
    // else
    //     std::cout << "The Y gradient is NOT correct!" << std::endl;

    // if constexpr (std::is_same_v<decltype(k), kernels::SquaredExpFull<Params>>) {
    //     // Generate symmetric matrices for the full squared exponential kernel
    //     Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim), B = Eigen::MatrixXd::Random(dim, dim), C(dim, dim), D(dim, dim);
    //     C = 0.5 * (A + A.transpose()) + dim * Eigen::MatrixXd::Identity(dim, dim);
    //     D = 0.5 * (B + B.transpose()) + dim * Eigen::MatrixXd::Identity(dim, dim);
    //     params_s = Eigen::Map<Eigen::VectorXd>(C.data(), C.size());
    //     params_t.segment(dim, dim * dim) = Eigen::Map<Eigen::VectorXd>(C.data(), C.size());
    //     Eigen::VectorXd v_t = Eigen::VectorXd::Random(k.sizeParams()), v_s = Eigen::VectorXd::Random(k.sizeParams() - 2);
    //     v_s = Eigen::Map<Eigen::VectorXd>(D.data(), D.size());
    //     v_t.segment(dim, dim * dim) = Eigen::Map<Eigen::VectorXd>(D.data(), D.size());

    //     if (checker.setDimension(k.sizeParams()).checkGradient(FunctionParamsT<dim>(), GradientParamsT<dim>(), params_t, v_t))
    //         std::cout << "TOTAL KERNEL: PARAMS gradient is CORRECT!" << std::endl;
    //     else
    //         std::cout << "TOTAL KERNEL: PARAMS gradient is NOT correct!" << std::endl;

    //     if (checker.setDimension(k.sizeParams() - 2).checkGradient(FunctionParamsS<dim>(), GradientParamsS<dim>(), params_s, v_s))
    //         std::cout << "SPECIFIC KERNEL: PARAMS gradient is CORRECT!" << std::endl;
    //     else
    //         std::cout << "SPECIFIC KERNEL: PARAMS gradient is NOT correct!" << std::endl;
    // }
    // else {
    //     if (checker.setDimension(k.sizeParams()).checkGradient(FunctionParamsT<dim>(), GradientParamsT<dim>()))
    //         std::cout << "TOTAL KERNEL: PARAMS gradient is CORRECT!" << std::endl;
    //     else
    //         std::cout << "TOTAL KERNEL: PARAMS gradient is NOT correct!" << std::endl;

    //     if (checker.setDimension(k.sizeParams() - 2).checkGradient(FunctionParamsS<dim>(), GradientParamsS<dim>()))
    //         std::cout << "SPECIFIC KERNEL: PARAMS gradient is CORRECT!" << std::endl;
    //     else
    //         std::cout << "SPECIFIC KERNEL: PARAMS gradient is NOT correct!" << std::endl;
    // }

    return 0;
}
