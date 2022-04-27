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

#include <kernel_lib/kernels/SquaredExp.hpp>
#include <kernel_lib/kernels/SquaredExpFull.hpp>

#define SQUAREDEXP kernels::SquaredExp<Params>
#define SQUAREDEXPFULL kernels::SquaredExpFull<Params>
#define KERNEL SQUAREDEXP

using namespace utils_lib;
using namespace kernel_lib;

struct Params {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0.405465);
        PARAM_SCALAR(double, sn, 0.693147);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, -0.356675);
    };

    struct exp_sq_full {
        PARAM_VECTOR(double, S, 1, 0.5, 0.5, 1);
    };
};

/* TOTAL KERNEL: Function in X */
template <int size>
struct FunctionX : public KERNEL {
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Zero(size);

    double operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return KERNEL()(x, y);
    }
};

/* TOTAL KERNEL: Gradient in X */
template <int size>
struct GradientX : public KERNEL {
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Zero(size);

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x) const
    {
        return KERNEL::grad(x, y);
    }
};

/* TOTAL KERNEL: Function in Y */
template <int size>
struct FunctionY : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

    double operator()(const Eigen::Matrix<double, size, 1>& y) const
    {
        return KERNEL()(x, y);
    }
};

/* TOTAL KERNEL: Gradient in Y */
template <int size>
struct GradientY : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& y) const
    {
        return KERNEL::grad(x, y, 0);
    }
};

/* TOTAL KERNEL: Hessian in X */
template <int size>
struct HessianXX : public KERNEL {
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Zero(size);

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x, const Eigen::Matrix<double, size, 1>& v) const
    {
        return KERNEL::hess(x, y, 0) * v;
    }
};

/* TOTAL KERNEL: Hessian in XY */
template <int size>
struct HessianXY : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& y, const Eigen::Matrix<double, size, 1>& v) const
    {
        return KERNEL::hess(x, y, 1) * v;
    }
};

/* TOTAL KERNEL: Hessian in YX */
template <int size>
struct HessianYX : public KERNEL {
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Zero(size);

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& x, const Eigen::Matrix<double, size, 1>& v) const
    {
        return KERNEL::hess(x, y, 2) * v;
    }
};

/* TOTAL KERNEL: Hessian in YY */
template <int size>
struct HessianYY : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);

    Eigen::Matrix<double, size, 1> operator()(const Eigen::Matrix<double, size, 1>& y, const Eigen::Matrix<double, size, 1>& v) const
    {
        return KERNEL::hess(x, y, 3) * v;
    }
};

/* TOTAL KERNEL: Function in PARAMS */
template <int size>
struct FunctionParamsT : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    double operator()(const Eigen::VectorXd& params)
    {
        KERNEL::setParams(params);

        return KERNEL::operator()(x, y);
    }
};

/* TOTAL KERNEL: Gradient in PARAMS */
template <int size>
struct GradientParamsT : public KERNEL {
    Eigen::Matrix<double, size, 1> x = Eigen::VectorXd::Zero(size);
    Eigen::Matrix<double, size, 1> y = Eigen::VectorXd::Ones(size);

    Eigen::VectorXd operator()(const Eigen::VectorXd& params)
    {
        KERNEL::setParams(params);

        return KERNEL::gradParams(x, y);
    }
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

    constexpr int dim = 2;

    Eigen::VectorXd x = Eigen::VectorXd::Random(dim), v = Eigen::VectorXd::Random(dim),
                    params_t = Eigen::VectorXd::Random(k.sizeParams()),
                    params_s = Eigen::VectorXd::Random(k.sizeParams() - 2);

    std::cout << "TOTAL KERNEL: Function in X test" << std::endl;
    std::cout << FunctionX<dim>()(x) << std::endl;

    std::cout << "TOTAL KERNEL: Gradient in X test" << std::endl;
    std::cout << GradientX<dim>()(x).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Function in Y test" << std::endl;
    std::cout << FunctionY<dim>()(x) << std::endl;

    std::cout << "TOTAL KERNEL: Gradient in Y test" << std::endl;
    std::cout << GradientY<dim>()(x).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Hessian in XX test" << std::endl;
    std::cout << HessianXX<dim>()(x, v).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Hessian in XY test" << std::endl;
    std::cout << HessianXY<dim>()(x, v).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Hessian in YX test" << std::endl;
    std::cout << HessianYX<dim>()(x, v).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Hessian in YY test" << std::endl;
    std::cout << HessianYY<dim>()(x, v).transpose() << std::endl;

    std::cout << "TOTAL KERNEL: Function in PARAMS" << std::endl;
    std::cout << FunctionParamsT<dim>()(params_t) << std::endl;

    std::cout << "TOTAL KERNEL:: Gradient in PARAMS" << std::endl;
    std::cout << GradientParamsT<dim>()(params_t).transpose() << std::endl;

    std::cout << "SPECIFIC KERNEL: Function in PARAMS" << std::endl;
    std::cout << FunctionParamsS<dim>()(params_s) << std::endl;

    std::cout << "SPECIFIC KERNEL: Gradient in PARAMS" << std::endl;
    std::cout << GradientParamsS<dim>()(params_s).transpose() << std::endl;

    DerivativeChecker checker(dim);

    if (checker.checkGradient(FunctionX<dim>(), GradientX<dim>()))
        std::cout << "The X gradient is CORRECT!" << std::endl;
    else
        std::cout << "The X gradient is NOT correct!" << std::endl;

    if (checker.checkGradient(FunctionY<dim>(), GradientY<dim>()))
        std::cout << "The Y gradient is CORRECT!" << std::endl;
    else
        std::cout << "The Y gradient is NOT correct!" << std::endl;

    if (checker.checkHessian(FunctionX<dim>(), GradientX<dim>(), HessianXX<dim>()))
        std::cout << "The XX hessian is CORRECT!" << std::endl;
    else
        std::cout << "The XX hessian is NOT correct!" << std::endl;

    if (checker.checkHessian(FunctionY<dim>(), GradientY<dim>(), HessianYY<dim>()))
        std::cout << "The YY hessian is CORRECT!" << std::endl;
    else
        std::cout << "The YY hessian is NOT correct!" << std::endl;

    if constexpr (std::is_same_v<decltype(k), kernels::SquaredExpFull<Params>>) {
        // Generate symmetric matrices for the full squared exponential kernel
        Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim), B = Eigen::MatrixXd::Random(dim, dim), C(dim, dim), D(dim, dim);
        C = 0.5 * (A + A.transpose()) + dim * Eigen::MatrixXd::Identity(dim, dim);
        D = 0.5 * (B + B.transpose()) + dim * Eigen::MatrixXd::Identity(dim, dim);
        params_s = Eigen::Map<Eigen::VectorXd>(C.data(), C.size());
        params_t.segment(dim, dim * dim) = Eigen::Map<Eigen::VectorXd>(C.data(), C.size());
        Eigen::VectorXd v_t = Eigen::VectorXd::Random(k.sizeParams()), v_s = Eigen::VectorXd::Random(k.sizeParams() - 2);
        v_s = Eigen::Map<Eigen::VectorXd>(D.data(), D.size());
        v_t.segment(dim, dim * dim) = Eigen::Map<Eigen::VectorXd>(D.data(), D.size());

        if (checker.setDimension(k.sizeParams()).checkGradient(FunctionParamsT<dim>(), GradientParamsT<dim>(), params_t, v_t))
            std::cout << "TOTAL KERNEL: PARAMS gradient is CORRECT!" << std::endl;
        else
            std::cout << "TOTAL KERNEL: PARAMS gradient is NOT correct!" << std::endl;

        if (checker.setDimension(k.sizeParams() - 2).checkGradient(FunctionParamsS<dim>(), GradientParamsS<dim>(), params_s, v_s))
            std::cout << "SPECIFIC KERNEL: PARAMS gradient is CORRECT!" << std::endl;
        else
            std::cout << "SPECIFIC KERNEL: PARAMS gradient is NOT correct!" << std::endl;
    }
    else {
        if (checker.setDimension(k.sizeParams()).checkGradient(FunctionParamsT<dim>(), GradientParamsT<dim>()))
            std::cout << "TOTAL KERNEL: PARAMS gradient is CORRECT!" << std::endl;
        else
            std::cout << "TOTAL KERNEL: PARAMS gradient is NOT correct!" << std::endl;

        if (checker.setDimension(k.sizeParams() - 2).checkGradient(FunctionParamsS<dim>(), GradientParamsS<dim>()))
            std::cout << "SPECIFIC KERNEL: PARAMS gradient is CORRECT!" << std::endl;
        else
            std::cout << "SPECIFIC KERNEL: PARAMS gradient is NOT correct!" << std::endl;
    }

    return 0;
}
