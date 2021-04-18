#ifndef KERNELLIB_KERNELS_MATERNTHREETWO_HPP
#define KERNELLIB_KERNELS_MATERNTHREETWO_HPP

#include "kernel_lib/kernels/AbstractKernel2.hpp"
#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {

    namespace defaults {
        struct matern_three_two {
            // Log length
            PARAM_VECTOR(double, l, 1, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class MaternThreeTwo : public AbstractKernel2<Params> {
        public:
            MaternThreeTwo()
            {
                AbstractKernel2<Params>::init();
            }

            /* Evaluate gradient */
            Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const
            {
                Eigen::MatrixXd grad;

                return grad;
            }

            /* Evaluate hessian */
            Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

            /* Evaluate parameters gradient */
            Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd grad;

                return grad;
            }

            /* Evaluate parameters hessian */
            Eigen::MatrixXd hessianParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd hess;

                return hess;
            }

        protected:
            double _temp;

            /* Kernel */
            Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd k;

                return k;
            }

            /* Set specific kernel parameters */
            void setParameters(const Eigen::VectorXd& params) {}

            /* Get number of parameters for the specific kernel */
            size_t sizeParameters() const { return 1; }
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_MATERNTHREETWO_HPP