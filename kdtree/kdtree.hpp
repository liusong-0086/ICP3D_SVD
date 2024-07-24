#ifndef KDTREE_H
#define KDTREE_H
#include <Eigen/Dense>
#include <iostream>
#include <stack>
#include <queue>
#include <numeric>
#include <execution>
#include <ostream>

struct KdTreeNode
{
    int idx_ = -1;
    size_t point_idx_ = 0;
    int split_axis_ = 0;
    double split_threval_ = 0.f;
    KdTreeNode *left_ = nullptr;
    KdTreeNode *right_ = nullptr;

    bool is_leaf() const { return left_ == nullptr && right_ == nullptr; };
};

struct NodeAndDistance
{
    NodeAndDistance(KdTreeNode *node, double distance) : node_(node), distance_(distance){};
    double distance_ = 0.f;
    KdTreeNode *node_ = nullptr;

    bool operator<(const NodeAndDistance &rhs) const { return distance_ < rhs.distance_; }
};

template <int dim>
class KdTree
{
public:
    using PointType = Eigen::Matrix<double, dim, 1>;

    explicit KdTree() = default;
    ~KdTree() { clear(); };

    // BFS build kdtree
    bool build(const std::vector<PointType> &cloud);

    // DFS search
    bool get_knn_points(const PointType &point, std::vector<size_t> &knn_idx, int k = 5);

    // MT DFS search
    bool get_knn_points_mt(const std::vector<PointType> &cloud,
                           std::vector<std::pair<size_t, size_t>> &matches, int k = 5);

    // DFS clear
    void clear();

    template <int dim>
    friend std::ostream &operator<<(std::ostream &out, const KdTree<dim> &tree);

private:
    void search_(const PointType &point, std::priority_queue<NodeAndDistance> &knn_result);

    bool need_prune_branches_(const PointType &point, KdTreeNode *node,
                              std::priority_queue<NodeAndDistance> &knn_result);

    void compute_split_parameters_(const std::vector<size_t> &points_index,
                                   double &thre, int &axis,
                                   std::vector<size_t> &left,
                                   std::vector<size_t> &right);

    void compute_leaf_distance(const PointType &point, KdTreeNode *node,
                               std::priority_queue<NodeAndDistance> &knn_result);

private:
    std::shared_ptr<KdTreeNode> m_root = nullptr;
    int m_k;
    std::vector<PointType> m_cloud;
};

template <int dim>
bool KdTree<dim>::build(const std::vector<PointType> &cloud)
{
    if (cloud.empty())
        return false;

    m_cloud = cloud;
    std::vector<size_t> cloud_idx(cloud.size());
    std::for_each(cloud_idx.begin(), cloud_idx.end(), [idx = 0](size_t &i) mutable
                  { i = idx++; });

    // Initalize root node
    m_root = std::make_shared<KdTreeNode>();
    std::queue<std::pair<KdTreeNode *, std::vector<size_t>>> que_;
    que_.push({m_root.get(), cloud_idx});

    while (!que_.empty())
    {
        int size = que_.size();

        for (int i = 0; i < size; ++i)
        {
            std::pair<KdTreeNode *, std::vector<size_t>> node_to_cloud = que_.front();
            que_.pop();

            KdTreeNode *node_temp = node_to_cloud.first;
            std::vector<size_t> cloud_idx_temp = node_to_cloud.second;

            std::vector<size_t> left_cloud_idx, right_cloud_idx;
            compute_split_parameters_(cloud_idx_temp, node_temp->split_threval_,
                                      node_temp->split_axis_,
                                      left_cloud_idx, right_cloud_idx);

            const auto create_leaf_node = [&que_](const std::vector<size_t> &index, KdTreeNode *&node)
            {
                if (!index.empty())
                {
                    node = new KdTreeNode;
                    if (1 == index.size()) // leaf node
                        node->point_idx_ = index[0];
                    else
                        que_.push({node, index});
                }
            };

            create_leaf_node(left_cloud_idx, node_temp->left_);
            create_leaf_node(right_cloud_idx, node_temp->right_);
        }
    }

    return true;
}

template <int dim>
bool KdTree<dim>::get_knn_points(const PointType &point, std::vector<size_t> &knn_idx, int k)
{
    m_k = k;

    std::priority_queue<NodeAndDistance> knn_result;
    search_(point, knn_result);

    knn_idx.clear();
    while (!knn_result.empty())
    {
        knn_idx.push_back(knn_result.top().node_->point_idx_);
        knn_result.pop();
    }

    return true;
}

template <int dim>
bool KdTree<dim>::get_knn_points_mt(const std::vector<PointType> &cloud,
                                    std::vector<std::pair<size_t, size_t>> &matches, int k)
{
    if (cloud.empty())
        return false;

    matches.resize(cloud.size() * k);

    std::vector<size_t> index(cloud.size());
    std::for_each(index.begin(), index.end(), [idx = 0](size_t &i) mutable
                  { i = idx++; });

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](size_t &i)
                  {
                    std::vector<size_t> knn_idx;
                    get_knn_points(cloud[i], knn_idx, k);
                    for (size_t j = 0; j < k; ++j)
                    {
                        matches[i * k + j].second = knn_idx[j];
                        if(j>=knn_idx.size())
                        {
                            matches[i * k + j].first = std::numeric_limits<size_t>::max();
                        }
                        else
                        {
                            matches[i * k + j].first = i;
                        }
                    } });

    return true;
}

template <int dim>
void KdTree<dim>::clear()
{
    std::stack<KdTreeNode *> st;

    if (m_root)
    {
        if (m_root->right_)
            st.push(m_root->right_);
        if (m_root->left_)
            st.push(m_root->left_);
    }

    while (!st.empty())
    {
        KdTreeNode *node_temp = st.top();
        st.pop();

        if (node_temp->right_)
            st.push(node_temp->right_);

        if (node_temp->left_)
            st.push(node_temp->left_);

        delete node_temp;
    }
}

template <int dim>
void KdTree<dim>::search_(const PointType &point, std::priority_queue<NodeAndDistance> &knn_result)
{
    std::stack<KdTreeNode *> st;
    st.push(m_root.get());

    while (!st.empty())
    {
        KdTreeNode *node_temp = st.top();
        st.pop();

        if (node_temp->is_leaf())
        {
            compute_leaf_distance(point, node_temp, knn_result);
        }
        else
        {
            KdTreeNode *this_side, *that_side;
            if (point[node_temp->split_axis_] <
                node_temp->split_threval_)
            {
                this_side = node_temp->left_;
                that_side = node_temp->right_;
            }
            else
            {
                this_side = node_temp->right_;
                that_side = node_temp->left_;
            }

            if (need_prune_branches_(point, node_temp, knn_result))
                if (that_side)
                    st.push(that_side);

            if (this_side)
                st.push(this_side);
        }
    }
}

template <int dim>
bool KdTree<dim>::need_prune_branches_(const PointType &point, KdTreeNode *node,
                                       std::priority_queue<NodeAndDistance> &knn_result)
{
    if (knn_result.size() < m_k)
        return true;

    double dist = point[node->split_axis_] - node->split_threval_;
    if (dist * dist < knn_result.top().distance_)
    {
        return true;
    }

    return false;
}

template <int dim>
void KdTree<dim>::compute_split_parameters_(const std::vector<size_t> &points_index,
                                            double &thre, int &axis,
                                            std::vector<size_t> &left,
                                            std::vector<size_t> &right)
{
    // Calculate mean
    Eigen::Matrix<double, dim, 1> means;
    means = std::accumulate(points_index.begin(), points_index.end(), Eigen::Matrix<double, dim, 1>::Zero().eval(),
                            [&](const Eigen::Matrix<double, dim, 1> &sum, const int &index)
                            { return sum + m_cloud[index]; }) /
            points_index.size();

    // Calculate variance
    Eigen::Matrix<double, dim, 1> vars;
    vars = std::accumulate(points_index.begin(), points_index.end(), Eigen::Matrix<double, dim, 1>::Zero().eval(),
                           [&](const Eigen::Matrix<double, dim, 1> &sum, const int &index)
                           { return sum + (m_cloud[index] - means).cwiseAbs2().eval(); }) /
           points_index.size();

    // Find the axis with maximum variance
    vars.maxCoeff(&axis);
    thre = means[axis];

    // Split point cloud to left and right
    std::for_each(points_index.begin(), points_index.end(), [&](const size_t &idx)
                  {
                    if (m_cloud[idx][axis] < thre)
                        left.emplace_back(idx);
                    else
                        right.emplace_back(idx); });
}

template <int dim>
void KdTree<dim>::compute_leaf_distance(const PointType &point, KdTreeNode *node,
                                        std::priority_queue<NodeAndDistance> &knn_result)
{
    double distance = (m_cloud[node->point_idx_] - point).squaredNorm();
    if (knn_result.size() < m_k)
    {
        knn_result.push({node, distance});
    }
    else
    {
        if (distance < knn_result.top().distance_)
        {
            knn_result.emplace(NodeAndDistance(node, distance));
            knn_result.pop();
        }
    }
}

template <int dim>
std::ostream &operator<<(std::ostream &out, const KdTree<dim> &tree)
{
    KdTreeNode *root = tree.m_root.get();
    std::queue<KdTreeNode *> que_;
    que_.push(root);

    while (!que_.empty())
    {
        int size = que_.size();

        for (int i = 0; i < size; ++i)
        {
            KdTreeNode *node = que_.front();
            que_.pop();

            if (node->is_leaf())
            {
                out << "Leaf Node index: " << node->point_idx_ << std::endl;
                out << "Point Cloud: " << std::endl
                    << tree.m_cloud[node->point_idx_] << std::endl;
            }

            if (node->left_)
                que_.push(node->left_);
            if (node->right_)
                que_.push(node->right_);
        }
    }

    return out;
}

#endif