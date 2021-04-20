#ifndef KERNELLIB_KERNELS_RIEMANNMATERN_HPP
#define KERNELLIB_KERNELS_RIEMANNMATERN_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {

    namespace defaults {
        struct riemann_matern {
            // Log length
            PARAM_VECTOR(double, l, 1, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params, typename EigenFunctions>
        class RiemannMatern : public AbstractKernel<Params> {
        public:
            RiemannMatern()
            {
                AbstractKernel<Params>::init();
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

#endif // KERNELLIB_KERNELS_RIEMANNMATERN_HPP