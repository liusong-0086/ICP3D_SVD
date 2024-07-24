#include "../common/common.h"
#include "kdtree.hpp"
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

void generateSphereCoordinates(std::vector<Eigen::Vector3d> &points, int numPoints)
{
    points.clear();
    double goldenRatio = (1 + std::sqrt(5.0)) / 2.0;
    double angleIncrement = CV_PI * 2 * goldenRatio;

    for (int i = 0; i < numPoints; ++i)
    {
        double t = double(i) / double(numPoints);
        double inclination = std::acos(1 - 2 * t);
        double azimuth = angleIncrement * i;

        double x = std::sin(inclination) * std::cos(azimuth);
        double y = std::sin(inclination) * std::sin(azimuth);
        double z = std::cos(inclination);

        points.push_back(Eigen::Vector3d(x, y, z));
    }
}

void transformToEllipsoid(std::vector<Eigen::Vector3d> &points, double a, double b, double c)
{
    for (auto &point : points)
    {
        point[0] *= a;
        point[1] *= b;
        point[2] *= c;
    }
}

std::vector<Eigen::Vector3d> generate_ellipsoid()
{
    // ellipsoid parameters
    double a = 200;
    double b = 100;
    double c = 150;

    int numPoints = 1000000;

    // Generate sphere coordinates and transform to ellipsoid
    std::vector<Eigen::Vector3d> spherePoints;
    generateSphereCoordinates(spherePoints, numPoints);
    transformToEllipsoid(spherePoints, a, b, c);

    return spherePoints;
}



void bf_nn(const std::vector<Eigen::Vector3d> &source,
           const std::vector<Eigen::Vector3d> &target,
           std::vector<std::pair<size_t, size_t>> &result)
{
    std::vector<size_t> index(target.size());
    std::for_each(index.begin(), index.end(), [idx = 0](size_t &i) mutable
                  { i = idx++; });

    const auto bfnn_points = [&source](const Eigen::Vector3d &p) -> int
    {
        return std::min_element(source.begin(), source.end(),
                                [&p](const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
                                { return (p1 - p).norm() < (p2 - p).norm(); }) -
               source.begin();
    };

    result.resize(target.size());
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](size_t &i)
                  {
        result[i].second = bfnn_points(target[i]);
        result[i].first = i; });
}

void test_kdtree3d()
{
    // Generate virtual point cloud
    std::vector<Eigen::Vector3d> points = generate_ellipsoid();
    Eigen::AngleAxisd rvec(CV_PI / 6, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d T(2, 2, 2);
    std::vector<Eigen::Vector3d> source(points.size());
    // std::transform(points.begin(), points.end(), source.begin(), [&rvec, &T](const Eigen::Vector3d &p)
    //                { return rvec.toRotationMatrix() * p + T; });

    source[0] = rvec.toRotationMatrix() * points[10] + T;
 
    // Build KD-tree
    KdTree<3> kd_tree;
    kd_tree.build(points);
    // std::cout << kd_tree << std::endl;
    // std::vector<std::pair<size_t, size_t>> matches;
    // kd_tree.get_knn_points_mt(points, matches, 1);
    std::vector<size_t> knn;
    kd_tree.get_knn_points(source[0], knn, 1);
    std::cout << "index : " << knn[0] << std::endl;
    // std::for_each(matches.begin(), matches.end(), [](const std::pair<size_t, size_t> &p_pair)
    //               { std::cout << "points index: " << p_pair.first << " source index: " << p_pair.second << std::endl; });
}

void test_bf_matches()
{
    // Generate virtual point cloud
    std::vector<Eigen::Vector3d> points = generate_ellipsoid();
    Eigen::AngleAxisd rvec(CV_PI / 6, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d T(2, 2, 2);
    std::vector<Eigen::Vector3d> source(points.size());
    std::transform(points.begin(), points.end(), source.begin(), [&rvec, &T](const Eigen::Vector3d &p)
                   { return rvec.toRotationMatrix() * p + T; });

    std::vector<std::pair<size_t, size_t>> bf_matches;
    bf_nn(source, points, bf_matches);
    // std::for_each(bf_matches.begin(), bf_matches.end(), [](const std::pair<size_t, size_t> &p_pair)
    //               { std::cout << "points index: " << p_pair.first << " source index: " << p_pair.second << std::endl; });
}

void test_pcl_kdtree()
{
    std::vector<Eigen::Vector3d> points = generate_ellipsoid();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Generate pointcloud data
    cloud->width = points.size();
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    for (std::size_t i = 0; i < cloud->size(); ++i)
    {
        (*cloud)[i].x = points[i][0];
        (*cloud)[i].y = points[i][1];
        (*cloud)[i].z = points[i][2];
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setEpsilon(0);
    kdtree.setInputCloud(cloud);
    pcl::PointXYZ searchPoint;

    int K = 1;
    std::vector<int> pointIdxKNNSearch(K);
    std::vector<float> pointKNNSquaredDistance(K);

    Eigen::AngleAxisd rvec(CV_PI / 6, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d T(2, 2, 2);
    Eigen::Vector3d p = points[10];
    p = rvec.toRotationMatrix() * p + T;

    pcl::PointXYZ p_search;
    p_search.x = p[0];
    p_search.y = p[1];
    p_search.z = p[2];

    kdtree.nearestKSearch(p_search, K, pointIdxKNNSearch, pointKNNSquaredDistance);
    std::cout<<"index : "<<pointIdxKNNSearch[0]<<std::endl;
}

void test_kdtree_bf_result()
{
    // Generate virtual point cloud
    std::vector<Eigen::Vector3d> points = generate_ellipsoid();
    Eigen::AngleAxisd rvec(CV_PI / 6, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d T(2, 2, 2);
    std::vector<Eigen::Vector3d> source(points.size());
    std::transform(points.begin(), points.end(), source.begin(), [&rvec, &T](const Eigen::Vector3d &p)
                   { return rvec.toRotationMatrix() * p + T; });

    // BF match
    std::vector<std::pair<size_t, size_t>> bf_matches;
    bf_nn(source, points, bf_matches);

    // Kdtree match
    KdTree<3> kd_tree;
    kd_tree.build(source);
    // std::cout << kd_tree << std::endl;
    std::vector<std::pair<size_t, size_t>> matches;
    kd_tree.get_knn_points_mt(points, matches, 1);

    int true_num = 0;
    for (size_t i = 0; i < bf_matches.size(); ++i)
    {
        if (bf_matches[i].second == matches[i].second)
        {
            true_num++;
        }
    }

    double true_p = double(true_num) / double(matches.size());
    std::cout << "True positive rate: " << true_p * 100 << std::endl;
}

int main()
{
    evaluate_and_call(test_pcl_kdtree, 1);
    evaluate_and_call(test_kdtree3d, 1);
    // evaluate_and_call(test_bf_matches, 1);
    // evaluate_and_call(test_kdtree_bf_result, 1);

    return 0;
}
