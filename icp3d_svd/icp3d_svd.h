#ifndef ICP3D_SVD_H
#define ICP3D_SVD_H
#include <Eigen/Dense>

class ICP3Dsvd
{
public:
    ICP3Dsvd();
    ~ICP3Dsvd() = default;
    bool setSourcePointCloud(const std::vector<Eigen::Vector3d> &source);
    bool setTargetPointCloud(const std::vector<Eigen::Vector3d> &target);
    bool registration(const double eps, const int iteratation,
                      Eigen::Matrix3d &R, Eigen::Vector3d &T);

private:
    class Impl;
    std::shared_ptr<Impl> m_pimpl;
};

#endif