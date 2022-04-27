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

#include <utils_lib/DerivativeChecker.hpp>
#include <utils_lib/FileManager.hpp>

#include <kernel_lib/kernels/RiemannMatern.hpp>
#include <kernel_lib/kernels/RiemannSqExp.hpp>
#include <kernel_lib/kernels/SquaredExp.hpp>

using namespace utils_lib;
using namespace kernel_lib;

// Kernel
#define EXPANSION utils::Expansion<ParamsEigenfunction, kernels::SquaredExp<ParamsEigenfunction>>
#define RIEMANNSQUAREDEXP kernels::RiemannSqExp<ParamsKernel, EXPANSION>
#define RIEMANNMATERN kernels::RiemannMatern<ParamsKernel, EXPANSION>

#define KERNEL RIEMANNMATERN

struct ParamsKernel {

    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.5);
        PARAM_SCALAR(double, sn, 1.4);
    };

    struct riemann_exp_sq : public defaults::riemann_exp_sq {
        PARAM_SCALAR(double, l, 1);
    };

    struct riemann_matern : public defaults::riemann_matern {
        PARAM_SCALAR(double, l, 1);

        PARAM_SCALAR(double, d, 2);

        PARAM_SCALAR(double, nu, 1.5);
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
struct GradientX {
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
struct FunctionY {
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
struct GradientY {
    GradientY(KERNEL& k) : _k(k)
    {
        _x.setOnes();
        _x.normalized();
    }

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return _k.grad(_x, x, 0);
    }

    Eigen::Matrix<double, size, 1> _x;
    KERNEL& _k;
};

/* TOTAL KERNEL: Hessian in X */
template <int size>
struct HessianXX {
    HessianXX(KERNEL& k) : _k(k)
    {
        _x.setOnes();
        _x.normalized();
    }

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x, const Eigen::Matrix<double, size, 1>& v) const
    {
        return _k.hess(x, _x, 0) * v;
    }

    Eigen::Matrix<double, size, 1> _x;
    KERNEL& _k;
};

/* TOTAL KERNEL: Hessian in XY */
template <int size>
struct HessianXY {
    HessianXY(KERNEL& k) : _k(k)
    {
        _x.setOnes();
        _x.normalized();
    }

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x, const Eigen::Matrix<double, size, 1>& v) const
    {
        return _k.hess(_x, x, 1) * v;
    }

    Eigen::Matrix<double, size, 1> _x;
    KERNEL& _k;
};

/* TOTAL KERNEL: Hessian in YX */
template <int size>
struct HessianYX {
    HessianYX(KERNEL& k) : _k(k)
    {
        _x.setOnes();
        _x.normalized();
    }

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x, const Eigen::Matrix<double, size, 1>& v) const
    {
        return _k.hess(x, _x, 2) * v;
    }

    Eigen::Matrix<double, size, 1> _x;
    KERNEL& _k;
};

/* TOTAL KERNEL: Hessian in YY */
template <int size>
struct HessianYY {
    HessianYY(KERNEL& k) : _k(k)
    {
        _x.setOnes();
        _x.normalized();
    }

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x, const Eigen::Matrix<double, size, 1>& v) const
    {
        return _k.hess(_x, x, 3) * v;
    }

    Eigen::Matrix<double, size, 1> _x;
    KERNEL& _k;
};

/* TOTAL KERNEL: Function in PARAMS */
template <int size>
struct FunctionParamsT {
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

        return _k(_x, _y);
    }

    Eigen::Matrix<double, size, 1> _x, _y;
    KERNEL& _k;
};

/* TOTAL KERNEL: Gradient in PARAMS */
template <int size>
struct GradientParamsT {
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

        return _k.gradParams(_x, _y);
    }

    Eigen::Matrix<double, size, 1> _x, _y;
    KERNEL& _k;
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
    Eigen::VectorXd v = Eigen::VectorXd::Random(dim);
    v = (Eigen::MatrixXd::Identity(x.rows(), x.rows()) - x * x.transpose()) * v;

    Eigen::VectorXd params_t = Eigen::VectorXd::Random(k.sizeParams()),
                    params_s = Eigen::VectorXd::Random(k.sizeParams() - 2);

    std::cout << "TOTAL KERNEL: Function in X test" << std::endl;
    FunctionX<dim> fx(k);
    std::cout << fx(x) << std::endl;

    std::cout << "TOTAL KERNEL: Gradient in X test" << std::endl;
    GradientX<dim> gx(k);
    std::cout << gx(x).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Function in Y test" << std::endl;
    FunctionY<dim> fy(k);
    std::cout << fy(x) << std::endl;

    std::cout << "TOTAL KERNEL: Gradient in Y test" << std::endl;
    GradientY<dim> gy(k);
    std::cout << gy(x).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Hessian in XX test" << std::endl;
    HessianXX<dim> hxx(k);
    std::cout << hxx(x, v).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Hessian in XY test" << std::endl;
    HessianXY<dim> hxy(k);
    std::cout << hxy(x, v).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Hessian in YX test" << std::endl;
    HessianYX<dim> hyx(k);
    std::cout << hyx(x, v).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Hessian in YY test" << std::endl;
    HessianYY<dim> hyy(k);
    std::cout << hyy(x, v).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Function in PARAMS" << std::endl;
    FunctionParamsT<dim> fpt(k);
    std::cout << fpt(params_t) << std::endl;

    std::cout << "TOTAL KERNEL:: Gradient in PARAMS" << std::endl;
    GradientParamsT<dim> gpt(k);
    std::cout << gpt(params_t).transpose() << std::endl;

    DerivativeChecker checker(dim);

    if (checker.checkGradient(fx, gx))
        std::cout << "The X gradient is CORRECT!" << std::endl;
    else
        std::cout << "The X gradient is NOT correct!" << std::endl;

    if (checker.checkGradient(fy, gy))
        std::cout << "The Y gradient is CORRECT!" << std::endl;
    else
        std::cout << "The Y gradient is NOT correct!" << std::endl;

    if (checker.checkHessian(fx, gx, hxx))
        std::cout << "The XX hessian is CORRECT!" << std::endl;
    else
        std::cout << "The XX hessian is NOT correct!" << std::endl;

    if (checker.checkHessian(fy, gy, hyy))
        std::cout << "The YY hessian is CORRECT!" << std::endl;
    else
        std::cout << "The YY hessian is NOT correct!" << std::endl;

    if (checker.setDimension(k.sizeParams()).checkGradient(fpt, gpt))
        std::cout << "TOTAL KERNEL: PARAMS gradient is CORRECT!" << std::endl;
    else
        std::cout << "TOTAL KERNEL: PARAMS gradient is NOT correct!" << std::endl;

    return 0;
}
