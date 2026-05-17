import numpy as np
import heapq
from sklearn.cluster import KMeans

# ==============================
# 参数配置
# ==============================

N = 10000
vector_dim = 64
scalar_dim = 5

Q = 100
topk = 10

leaf_size = 200
ivf_clusters = 8
probe_clusters = 4

np.random.seed(999)

# ==============================
# 数据生成
# ==============================

vector_data = np.random.randn(N, vector_dim).astype(np.float32)
scalar_data = np.random.rand(N, scalar_dim).astype(np.float32)

query_vectors = np.random.randn(Q, vector_dim).astype(np.float32)
query_ranges = [(0.2, 0.8)] * scalar_dim
query_scalar = np.array([(l + r) / 2 for l, r in query_ranges])


# ==============================
# KDNode 类 (方案1: 均值路由)
# ==============================


class KDNode:
    def __init__(self, indices, depth=0):
        self.left = None
        self.right = None

        # KD-Tree 分割信息
        self.axis = depth % scalar_dim
        self.is_leaf = len(indices) <= leaf_size

        # 标量包围盒 (用于范围剪枝)
        self.bbox_min = np.min(scalar_data[indices], axis=0)
        self.bbox_max = np.max(scalar_data[indices], axis=0)

        # 向量空间统计 (方案1核心: 路由 + 剪枝)
        all_vectors = vector_data[indices].astype(np.float32)
        self.mean_vector = np.mean(all_vectors, axis=0)
        self.max_radius = np.max(np.linalg.norm(all_vectors - self.mean_vector, axis=1))

        if self.is_leaf:
            # 🍃 叶子节点: 存储实际数据索引
            self.indices = indices
            self.split = None
        else:
            # 🌲 非叶子节点: 不存indices省内存，递归构建子树
            self.indices = None

            # 按标量维度分割
            sorted_idx = indices[np.argsort(scalar_data[indices][:, self.axis])]
            median = len(sorted_idx) // 2
            self.split = scalar_data[sorted_idx[median]][self.axis]

            self.left = KDNode(sorted_idx[:median], depth + 1)
            self.right = KDNode(sorted_idx[median:], depth + 1)

            # ❌ 删除: node_centroids (方案1不需要聚类中心)


print("Building KDTree (Scheme 1: Mean-Vector Routing)...")
root = KDNode(np.arange(N))
print("✓ KDTree built")


# ==============================
# 收集叶子节点
# ==============================

leaf_nodes = []


def collect_leaves(node):
    if node.is_leaf:
        node.leaf_id = len(leaf_nodes)
        leaf_nodes.append(node)
    else:
        collect_leaves(node.left)
        collect_leaves(node.right)


collect_leaves(root)
print(f"✓ Collected {len(leaf_nodes)} leaf nodes")


# ==============================
# 构建叶子节点 IVF 索引
# ==============================

ivf_index = {}

print("Building IVF in leaves...")

for node in leaf_nodes:
    lid = node.leaf_id
    indices = node.indices
    vectors = vector_data[indices].astype(np.float32)

    if len(indices) < ivf_clusters:
        ivf_index[lid] = {
            "centroids": None,
            "labels": None,
            "vectors": vectors,
        }
        continue

    kmeans = KMeans(n_clusters=ivf_clusters, random_state=42, n_init=5, max_iter=100)
    kmeans.fit(vectors)

    ivf_index[lid] = {
        "centroids": kmeans.cluster_centers_,
        "labels": kmeans.labels_,
        "vectors": vectors,
    }

print("✓ IVF built")


# ==============================
# 工具函数
# ==============================


def scalar_match(row):
    """标量属性范围过滤"""
    for d in range(scalar_dim):
        if row[d] < query_ranges[d][0] or row[d] > query_ranges[d][1]:
            return False
    return True


def bbox_intersect(node):
    """节点包围盒与查询范围相交判断"""
    for d in range(scalar_dim):
        if (
            node.bbox_max[d] < query_ranges[d][0]
            or node.bbox_min[d] > query_ranges[d][1]
        ):
            return False
    return True


def prune_by_node(node, query_vector, heap):
    """向量空间三角不等式粗剪"""
    if len(heap) < topk:
        return False
    dist_to_mean = np.linalg.norm(query_vector - node.mean_vector)
    if dist_to_mean - node.max_radius > -heap[0][0]:
        return True
    return False


def update_heap(heap, dist, idx):
    """维护最大堆存储 topk"""
    if len(heap) < topk:
        heapq.heappush(heap, (-dist, idx))
    elif dist < -heap[0][0]:
        heapq.heapreplace(heap, (-dist, idx))


# ==============================
# 查询函数 (方案1: 均值路由) - 🔧 修复版
# ==============================


def hybrid_query(query_vector):
    heap = []
    query_vector = np.array(query_vector, dtype=np.float32)

    def recurse(node):
        # 1. 标量剪枝
        if not bbox_intersect(node):
            return

        # 2. 向量粗剪
        if prune_by_node(node, query_vector, heap):
            return

        if node.is_leaf:
            # 🍃 叶子节点: 搜索 IVF
            _search_leaf(node, query_vector, heap)
        else:
            # 🌲 非叶子节点: 🔧 方案1均值路由 (内联修复)
            # 计算 query 到左右子树均值的距离
            dist_left = np.linalg.norm(query_vector - node.left.mean_vector)
            dist_right = np.linalg.norm(query_vector - node.right.mean_vector)

            # 优先搜索更近的子树
            if dist_left <= dist_right:
                primary, secondary = node.left, node.right
            else:
                primary, secondary = node.right, node.left

            # 递归搜索
            recurse(primary)
            if not prune_by_node(secondary, query_vector, heap):
                recurse(secondary)

    recurse(root)
    return sorted([idx for _, idx in heap])


def _search_leaf(node, query_vector, heap):
    """叶子节点搜索 (IVF)"""
    leaf_info = ivf_index[node.leaf_id]
    indices = node.indices
    vectors = leaf_info["vectors"]
    centroids = leaf_info["centroids"]
    labels = leaf_info["labels"]

    if centroids is None:
        # 数据少: 暴力扫描
        for i, idx in enumerate(indices):
            if not scalar_match(scalar_data[idx]):
                continue
            dist = np.linalg.norm(vectors[i] - query_vector)
            update_heap(heap, dist, idx)
        return

    # IVF: 只搜索最近的 probe_clusters 个簇
    centroid_dists = np.linalg.norm(centroids - query_vector, axis=1)
    nearest_cids = np.argsort(centroid_dists)[: min(probe_clusters, len(centroids))]

    for cid in nearest_cids:
        mask = labels == cid
        local_ids = np.where(mask)[0]
        for lid in local_ids:
            idx = indices[lid]
            if not scalar_match(scalar_data[idx]):
                continue
            dist = np.linalg.norm(vectors[lid] - query_vector)
            update_heap(heap, dist, idx)


# ==============================
# Ground Truth (暴力验证)
# ==============================

print("Computing ground truth...")
ground_truth = []

for q in query_vectors:
    dists = []
    for i in range(N):
        if not scalar_match(scalar_data[i]):
            continue
        dist = np.linalg.norm(vector_data[i] - q)
        dists.append((dist, i))
    dists.sort()
    ground_truth.append([idx for _, idx in dists[:topk]])

print("✓ Ground truth ready")


# ==============================
# Recall 评估
# ==============================

print("Evaluating recall...")
total_recall = 0

for qi, q in enumerate(query_vectors):
    approx = hybrid_query(q)
    gt = set(ground_truth[qi])
    hit = len(set(approx) & gt)
    total_recall += hit / topk

recall = total_recall / Q

print(f"\n📊 Recall@{topk}: {recall:.4f}")
print(f"   Scheme: Mean-Vector Routing (no node_centroids)")
