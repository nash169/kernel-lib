#include <iostream>
#include <kernel_lib/Kernel.hpp>

using namespace kernel_lib;

int main(int argc, char const* argv[])
{
    utils::FileManager io_manager;

    // Load data
    Eigen::MatrixXd X = io_manager.read<Eigen::MatrixXd>("rsc/icub/joints_icub.csv"),
                    V = io_manager.read<Eigen::MatrixXd>("rsc/icub/vels_icub.csv");

    Eigen::VectorXd dt = io_manager.read<Eigen::MatrixXd>("rsc/icub/dtime_icub.csv");

    // Reduce dimension
    size_t dim = 15, num_samples = X.rows();

    Eigen::MatrixXd X_red = X.block(0, 0, num_samples, dim),
                    V_red = V.block(0, 0, num_samples, dim);

    // Normalize
    Eigen::VectorXd X_mean = X_red.colwise().mean(), V_mean = V_red.colwise().mean();
    Eigen::VectorXd X_std = Eigen::VectorXd::Zero(dim), V_std = Eigen::VectorXd::Zero(dim);

    for (size_t i = 0; i < num_samples; i++) {
        X_std = X_std.array() + (X_red.row(i).transpose() - X_mean).array().pow(2) / (num_samples - 1);
        V_std = V_std.array() + (V_red.row(i).transpose() - V_mean).array().pow(2) / (num_samples - 1);
    }

    X_std = X_std.array().sqrt();
    V_std = V_std.array().sqrt();

    for (size_t i = 0; i < num_samples; i++) {
        X_red.row(i) = (X_red.row(i) - X_mean.transpose()).array() / X_std.transpose().array();
        V_red.row(i) = (V_red.row(i) - V_mean.transpose()).array() / V_std.transpose().array();
    }

    // Dataset
    Eigen::MatrixXd Dataset(num_samples, 2 * dim);
    Dataset << X_red, V_red;

    // Calculate sigma and angle reference
    double scale = 2.5, max_d = (V_red.rowwise().norm().array() * dt.array()).maxCoeff(),
           sigma = scale * max_d,
           angle_ref = M_PI / 6;

    // Velocity Directed Kernel
    ExpVelocityDirected k;
    k.setAngle(angle_ref).setParams(tools::makeVector(sigma));

    // Exp Kernel
    // Exp k;
    // Eigen::VectorXd params(1);
    // params << sigma;
    // k.setParams(params);

    Eigen::MatrixXd K = k(Dataset, Dataset).reshaped(num_samples, num_samples);
    io_manager.setFile("rsc/eigensystem/gramian.csv");
    io_manager.write("gramian", K);

    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
    // io_manager.setFile("rsc/eigensystem/eigendata.csv");
    // io_manager.write("eigenvalues", es.eigenvalues());
    // io_manager.append("eigenvec_1", es.eigenvectors().col(num_samples - 1), "eigenvec_2", es.eigenvectors().col(num_samples - 2), "eigenvec_3", es.eigenvectors().col(num_samples - 3));

    return 0;
}