#define USE_CPP17_MT

#include "rigid_transform.hpp"
#include "../common/common.h"

void test_2Drigid_transform()
{
    std::vector<Eigen::Vector2d> P;
    P.push_back(Eigen::Vector2d(100, 0));
    P.push_back(Eigen::Vector2d(0, 100));

    Eigen::Matrix2d R;
    double theta = CV_PI / 6;
    R << std::cos(theta), -std::sin(theta),
        std::sin(theta), std::cos(theta);

    Eigen::Vector2d T(10, 10);

    std::vector<Eigen::Vector2d> Q;
    for (const auto &p : P)
    {
        Eigen::Vector2d transformed = R * p + T;
        Q.push_back(transformed);
    }

    RigidTransform<2> rigid_transform_2d;
    Eigen::Matrix3d SE2 = rigid_transform_2d.solve(P, Q);
    std::cout << "SE2: " << std::endl;
    std::cout << SE2 << std::endl;
}

void test_3Drigid_transform()
{
    std::vector<Eigen::Vector3d> P;
    P.push_back(Eigen::Vector3d(100, 0, 0));
    P.push_back(Eigen::Vector3d(0, 100, 0));
    P.push_back(Eigen::Vector3d(0, 0, 100));

    Eigen::AngleAxisd rotateX(CV_PI / 6, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d T(0, 0, 0);

    std::vector<Eigen::Vector3d> Q;
    for (const auto &p : P)
    {
        Eigen::Vector3d transformed = rotateX.matrix() * p + T;
        Q.push_back(transformed);
    }

    RigidTransform<3> rigid_transform_3d;
    Eigen::Matrix4d SE3 = rigid_transform_3d.solve(P, Q);

    std::cout << "SE3: " << std::endl;
    std::cout << SE3 << std::endl;
}

int main()
{
    std::cout << "================3D================" << std::endl;
    evaluate_and_call(test_3Drigid_transform, 1);
    std::cout << "================2D================" << std::endl;
    evaluate_and_call(test_2Drigid_transform, 1);

    return 0;
}