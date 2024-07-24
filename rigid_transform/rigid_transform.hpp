#ifndef RIGID_TRANSFORM_H
#define RIGID_TRANSFORM_H
#include <Eigen/Dense>
#include <execution>

template <int dim>
class RigidTransform
{
public:
    using PointType = Eigen::Matrix<double, dim, 1>;
    using RigidTransformType = Eigen::Matrix<double, dim + 1, dim + 1>;

    RigidTransform() = default;
    ~RigidTransform() = default;
    RigidTransformType solve(const std::vector<PointType> &P,
                             const std::vector<PointType> &Q);
    RigidTransformType solve(const Eigen::Matrix<double, dim, dim> &M,
                             const Eigen::Matrix<double, dim, 1> &P_center,
                             const Eigen::Matrix<double, dim, 1> &Q_center);

private:
    PointType normalize_point_(std::vector<PointType> &points);
    PointType normalize_point_mt_(std::vector<PointType> &points);
    Eigen::Matrix<double, dim, dim> compute_M(const std::vector<PointType> &X,
                                              const std::vector<PointType> &Y);
    Eigen::Matrix<double, dim, dim> compute_M_mt(const std::vector<PointType> &X,
                                                 const std::vector<PointType> &Y);
};

template <int dim>
Eigen::Matrix<double, dim + 1, dim + 1> RigidTransform<dim>::solve(const std::vector<PointType> &P,
                                                                   const std::vector<PointType> &Q)
{
    if (P.size() != Q.size() || P.size() < dim)
    {
        throw std::runtime_error("point cloud size must be >= 3 and source size == target size!");
    }

    auto X = P;
    auto Y = Q;

#ifdef USE_CPP17_MT
    // Point cloud normalize multi thread
    PointType P_center = normalize_point_mt_(X);
    PointType Q_center = normalize_point_mt_(Y);
    // Compute M multi thread
    Eigen::Matrix<double, dim, dim> S = compute_M_mt(X, Y);
#else
    // Point cloud normalize single thread
    PointType P_center = normalize_point_(X);
    PointType Q_center = normalize_point_(Y);
    // Compute M single thread
    Eigen::Matrix<double, dim, dim> S = compute_M(X, Y);
#endif

    return solve(S, P_center, Q_center);
}

template <int dim>
Eigen::Matrix<double, dim + 1, dim + 1> RigidTransform<dim>::solve(const Eigen::Matrix<double, dim, dim> &S,
                                                                   const Eigen::Matrix<double, dim, 1> &P_center,
                                                                   const Eigen::Matrix<double, dim, 1> &Q_center)
{
    // Compute Rotation
    Eigen::JacobiSVD<Eigen::Matrix<double, dim, dim>> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, dim, dim> U = svd.matrixU();
    Eigen::Matrix<double, dim, dim> V = svd.matrixV();

    Eigen::Matrix<double, dim, dim> D = Eigen::Matrix<double, dim, dim>::Identity();
    D(dim - 1, dim - 1) = (V * U.transpose()).determinant();
    Eigen::Matrix<double, dim, dim> R = V * D * U.transpose();

    // Compute Translation
    PointType T = Q_center - R * P_center;

    // Construct SE(3)
    RigidTransformType SE3 = RigidTransformType::Identity();
    SE3.block<dim, dim>(0, 0) = R;
    SE3.block<dim, 1>(0, dim) = T;

    return SE3;
}

template <int dim>
Eigen::Matrix<double, dim, 1> RigidTransform<dim>::normalize_point_(std::vector<PointType> &points)
{
    // Get point cloud center
    PointType center = PointType::Zero();
    for (size_t i = 0; i < points.size(); ++i)
    {
        center += points[i];
    }
    center = center / points.size();

    // Remove center
    for (size_t i = 0; i < points.size(); ++i)
    {
        points[i] = points[i] - center;
    }

    return center;
}

template <int dim>
Eigen::Matrix<double, dim, 1> RigidTransform<dim>::normalize_point_mt_(std::vector<PointType> &points)
{
    // Get point cloud center
    PointType center = std::reduce(points.begin(), points.end(), PointType::Zero().eval(),
                                   [](const PointType &sum, const PointType &item)
                                   { return sum + item; }) /
                       points.size();

    // Remove center
    std::transform(points.begin(), points.end(), points.begin(), [&center](const PointType &p)
                   { return p - center; });

    return center;
}

template <int dim>
Eigen::Matrix<double, dim, dim> RigidTransform<dim>::compute_M(const std::vector<PointType> &X,
                                                               const std::vector<PointType> &Y)
{
    Eigen::Matrix<double, dim, dim> M = Eigen::Matrix<double, dim, dim>::Zero();

    for (size_t i = 0; i < X.size(); ++i)
    {
        M += X[i] * Y[i].transpose();
    }

    return M;
}

template <int dim>
Eigen::Matrix<double, dim, dim> RigidTransform<dim>::compute_M_mt(const std::vector<PointType> &X,
                                                                  const std::vector<PointType> &Y)
{
    std::vector<size_t> index(X.size());
    std::for_each(index.begin(), index.end(), [idx = 0](size_t &i) mutable
                  { i = idx++; });
    std::vector<Eigen::Matrix<double, dim, dim>> M_vec(index.size());
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i)
                  { M_vec[i] = X[i] * Y[i].transpose(); });

    return std::reduce(M_vec.begin(), M_vec.end(), Eigen::Matrix<double, dim, dim>::Zero().eval(),
                           [](const Eigen::Matrix<double, dim, dim> &sum, const Eigen::Matrix<double, dim, dim> &item)
                           { return sum + item; });
}

#endif