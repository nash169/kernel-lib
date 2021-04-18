#ifndef KERNELLIB_KERNELS_ABSTRACTKERNEL2_HPP
#define KERNELLIB_KERNELS_ABSTRACTKERNEL2_HPP

#include "kernel_lib/tools/macros.hpp"
#include "kernel_lib/tools/math.hpp"

namespace kernel_lib {

    namespace defaults {
        struct kernel2 {
            // Log signal std
            PARAM_SCALAR(double, sf, 1.0);

            // Log noise std
            PARAM_SCALAR(double, sn, 0.0);
        };
    } // namespace defaults

    namespace kernels {
        template <typename Params>
        class AbstractKernel2 {
        public:
            AbstractKernel2() : _sf2(std::exp(2 * Params::kernel2::sf())), _sn2(std::exp(2 * Params::kernel2::sn())) {}

            /* Evaluate kernel */
            Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const { return this->kernel(x, y); }

            /* Evaluate gradient */
            virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const = 0;

            /* Evaluate hessian */
            virtual Eigen::MatrixXd hessian(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const size_t& i) const = 0;

            /* Evaluate parameters gradient */
            virtual Eigen::MatrixXd gradientParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;

            /* Evaluate parameters hessian */
            virtual Eigen::MatrixXd hessianParams(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;

            /* Parameters */
            Eigen::VectorXd params() const
            {
                return _params;
            }

            /* Set parameters */
            void setParams(const Eigen::VectorXd& params)
            {
                _params(0) = params(0);
                _params(1) = params(1);

                _sf2 = std::exp(2 * params(0));
                _sn2 = std::exp(2 * params(1));

                this->setParameters(params.segment(2, params.rows() - 2));
            }

            /* Parameters' size */
            size_t sizeParams() const { return this->sizeParameters() + 2; }

        protected:
            double _sf2, _sn2;

            Eigen::VectorXd _params;

            // Init
            void init()
            {
                _params(0) = Params::kernel2::sf();
                _params(1) = Params::kernel2::sn();
            }

            /* Kernel */
            virtual Eigen::MatrixXd kernel(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;

            /* Set specific kernel parameters */
            virtual void setParameters(const Eigen::VectorXd& params) = 0;

            /* Get number of parameters for the specific kernel */
            virtual size_t sizeParameters() const = 0;
        };

    } // namespace kernels
} // namespace kernel_lib

#endif // KERNELLIB_KERNELS_ABSTRACTKERNEL2_HPP