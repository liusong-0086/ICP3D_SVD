#include "icp3d_svd.h"

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include "../common/common.h"

void test_pcl_icp()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::string path = R"(.\testdata\rabbit.pcd)";
    pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud);

    Eigen::Isometry3f SE3 = Eigen::Isometry3f::Identity();
    Eigen::AngleAxisf rvec(M_PI / 18, Eigen::Vector3f::UnitZ());
    Eigen::Vector3f T(0.005, 0.005, 0.005);
    SE3.pretranslate(T);
    SE3.rotate(rvec);

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *transformed_cloud, SE3.matrix());

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud);
    icp.setInputTarget(transformed_cloud);
    icp.setEuclideanFitnessEpsilon(0.0001);
    icp.setMaximumIterations(100);

    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);

    std::cout << "SE3: " << std::endl
              << icp.getFinalTransformation() << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "True: " << std::endl;
    std::cout << "R: " << rvec.matrix() << std::endl;
    std::cout << "T: " << T.transpose() << std::endl;
}

void test_icp_svd()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::string path = R"(.\testdata\rabbit.pcd)";
    pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud);

    std::vector<Eigen::Vector3d> cloud_eigen;
    for (const auto &pt : cloud->points)
    {
        cloud_eigen.push_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
    }

    Eigen::AngleAxisd rvec(M_PI / 18, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d T(0.005, 0.005, 0.005);
    std::vector<Eigen::Vector3d> cloud_eigen2(cloud_eigen.size());
    std::transform(cloud_eigen.begin(), cloud_eigen.end(), cloud_eigen2.begin(), [&](const Eigen::Vector3d &p)
                   { return rvec * p + T; });

    ICP3Dsvd icp;
    icp.setSourcePointCloud(cloud_eigen);
    icp.setTargetPointCloud(cloud_eigen2);
    
    Eigen::Matrix3d R_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d T_ = Eigen::Vector3d::Zero();
    icp.registration(0.0001, 100, R_, T_);
    std::cout << "R: " << std::endl
              << R_ << std::endl
              << "T: " << T_.transpose() << std::endl;

    std::cout << "------------------------------" << std::endl;
    std::cout << "True: " << std::endl;
    std::cout << "R: " << rvec.matrix() << std::endl;
    std::cout << "T: " << T.transpose() << std::endl;
}

int main()
{
    std::cout << "PCL=================================" << std::endl;
    evaluate_and_call(test_pcl_icp, 1);
    std::cout << "ICP SVD=============================" << std::endl;
    evaluate_and_call(test_icp_svd, 1);

    return 0;
}
