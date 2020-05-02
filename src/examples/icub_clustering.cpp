#include <iostream>
#include <kernel_lib/Kernel.hpp>

using namespace kernel_lib;

int main(int argc, char const* argv[])
{
    utils::FileManager io_manager;

    Eigen::MatrixXd X = io_manager.read<Eigen::MatrixXd>("rsc/icub/joints_icub.csv"),
                    V = io_manager.read<Eigen::MatrixXd>("rsc/icub/vels_icub.csv"),
                    Dataset(X.rows(), X.cols() + V.cols());

    Dataset << X, V;

    Eigen::VectorXd labels = io_manager.read<Eigen::MatrixXd>("rsc/icub/labels_icub.csv"),
                    dtimes = io_manager.read<Eigen::MatrixXd>("rsc/icub/dtime_icub.csv");

    ExpVelocityDirected exp_velocity;
    Eigen::MatrixXd K = exp_velocity(Dataset, Dataset).reshaped(X.rows(), X.rows());

    Eigen::EigenSolver<Eigen::MatrixXd> es(K);

    // io_manager.setFile("rsc/kernel_eval/exp_velocity.csv");
    // io_manager.write("Gram Matrix", K);

    return 0;
}