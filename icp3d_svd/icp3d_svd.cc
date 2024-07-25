#include <execution>

#include "icp3d_svd.h"
#include "../kdtree/kdtree.hpp"
#define USE_CPP17_MT
#include "../rigid_transform/rigid_transform.hpp"

class ICP3Dsvd::Impl
{
public:
    Impl()
    {
        m_p_Kdtree = std::make_unique<KdTree<3>>();
        m_p_estimator = std::make_unique<RigidTransform<3>>();
    };
    ~Impl() = default;
    bool setSourcePointCloud(const std::vector<Eigen::Vector3d> &source);
    bool setTargetPointCloud(const std::vector<Eigen::Vector3d> &target);
    bool registration(const double eps, const int iteration,
                      Eigen::Matrix3d &R, Eigen::Vector3d &T);

private:
    void transformPointCloud_(const Eigen::Matrix3d &R, const Eigen::Vector3d &T,
                              std::vector<Eigen::Vector3d> &point_cloud);

    double matchPointCloud_(std::vector<Eigen::Vector3d> &source);

private:
    std::unique_ptr<KdTree<3>> m_p_Kdtree;
    std::unique_ptr<RigidTransform<3>> m_p_estimator;
    std::vector<Eigen::Vector3d> m_source, m_target;
};

bool ICP3Dsvd::Impl::setSourcePointCloud(const std::vector<Eigen::Vector3d> &source)
{
    if (source.empty())
        return false;

    m_source = source;

    return true;
}

bool ICP3Dsvd::Impl::setTargetPointCloud(const std::vector<Eigen::Vector3d> &target)
{
    if (target.empty())
        return false;

    m_target = target;

    m_p_Kdtree->build(m_target);
    return true;
}

bool ICP3Dsvd::Impl::registration(const double eps, const int iteration,
                                  Eigen::Matrix3d &R, Eigen::Vector3d &T)
{
    // 1. Initialize RigidTransformation
    transformPointCloud_(R, T, m_source);
    Eigen::Matrix4d final_transformation = Eigen::Matrix4d::Identity();
    final_transformation.block<3, 3>(0, 0) = R;
    final_transformation.block<3, 1>(0, 3) = T;

    for (int i = 0; i < iteration; ++i)
    {
        // 2. Knn search
        std::vector<Eigen::Vector3d> source_;
        double err_ = matchPointCloud_(source_);
        // 3. Estimate RigidTransformation
        Eigen::Matrix4d cur_transformation = m_p_estimator->solve(source_, m_target);
        // 4. Update Point Cloud
        transformPointCloud_(cur_transformation.block<3, 3>(0, 0),
                             cur_transformation.block<3, 1>(0, 3), m_source);
        // 5. Update Transformation
        final_transformation = cur_transformation * final_transformation;

        // 6. Converage
        if (err_ < eps)
        {
            R = final_transformation.block<3, 3>(0, 0);
            T = final_transformation.block<3, 1>(0, 3);
            return true;
        }
    }

    return false;
}

void ICP3Dsvd::Impl::transformPointCloud_(const Eigen::Matrix3d &R, const Eigen::Vector3d &T,
                                          std::vector<Eigen::Vector3d> &point_cloud)
{
    std::transform(point_cloud.begin(), point_cloud.end(), point_cloud.begin(),
                   [&](const Eigen::Vector3d &p)
                   { return R * p + T; });
}

double ICP3Dsvd::Impl::matchPointCloud_(std::vector<Eigen::Vector3d> &source)
{
    std::vector<std::pair<size_t, size_t>> match_indexs;
    m_p_Kdtree->get_knn_points_mt(m_source, match_indexs, 1);

    source.resize(match_indexs.size());
    double err = 0.f;
    std::for_each(match_indexs.begin(), match_indexs.end(),
                  [&source, this, &err](const std::pair<size_t, size_t> &item)
                  {
                      source[item.first] = m_source[item.second];
                      err += (m_source[item.second] - m_target[item.first]).norm();
                  });

    return err / match_indexs.size();
}

ICP3Dsvd::ICP3Dsvd()
{
    m_pimpl = std::make_unique<Impl>();
}

bool ICP3Dsvd::setSourcePointCloud(const std::vector<Eigen::Vector3d> &source)
{
    return m_pimpl->setSourcePointCloud(source);
}

bool ICP3Dsvd::setTargetPointCloud(const std::vector<Eigen::Vector3d> &target)
{
    return m_pimpl->setTargetPointCloud(target);
}

bool ICP3Dsvd::registration(const double eps, const int iteration,
                            Eigen::Matrix3d &R, Eigen::Vector3d &T)
{
    return m_pimpl->registration(eps, iteration, R, T);
}