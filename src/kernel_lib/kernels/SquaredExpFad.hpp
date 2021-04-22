#ifndef KERNELLIB_KERNELS_SQUAREDEXPFAD_HPP
#define KERNELLIB_KERNELS_SQUAREDEXPFAD_HPP

#include "kernel_lib/kernels/AbstractKernel.hpp"
#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

/*
For implemention check:
https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian
https://math.stackexchange.com/questions/2867022/derivation-of-derivative-of-multivariate-gaussian-w-r-t-covariance-matrix
https://stats.stackexchange.com/questions/27436/how-to-take-derivative-of-multivariate-normal-density
*/

namespace kernel_lib {

    namespace defaults {
        struct exp_sq_fad {
            // Log length
            PARAM_VECTOR(double, l, 1, 1);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class SquaredExpFad : public AbstractKernel<Params> { // Factor Analysis Distance
        public:
            SquaredExpFad()
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

#endif // KERNELLIB_KERNELS_SQUAREDEXPFAD_HPP