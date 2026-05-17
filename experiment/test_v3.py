import numpy as np
import heapq
from sklearn.cluster import KMeans

# ==============================
# 参数
# ==============================
N = 10000
vector_dim = 64
scalar_dim = 5
Q = 100
topk = 10
leaf_size = 200
ivf_clusters = 8
probe_clusters = 4

np.random.seed(42)

# ==============================
# 1. 生成数据
# ==============================
vector_data = np.random.randn(N, vector_dim)
scalar_data = np.random.rand(N, scalar_dim)

query_vectors = np.random.randn(Q, vector_dim)
query_ranges = [(0.2, 0.8)] * scalar_dim


# ==============================
# 2. 自定义 KDTree (带内部节点剪枝)
# ==============================
class KDNode:
    def __init__(self, indices, depth=0):
        self.indices = indices
        self.left = None
        self.right = None
        self.axis = depth % scalar_dim
        self.split = None
        self.bbox_min = np.min(scalar_data[indices], axis=0)
        self.bbox_max = np.max(scalar_data[indices], axis=0)

        self.is_leaf = len(indices) <= leaf_size
        self.node_centroids = None  # 内部节点聚类中心，用于粗剪
        self.mean_vector = None  # 内部节点均值向量
        self.max_radius = None  # 内部节点最大向量半径

        if self.is_leaf:
            # 叶子节点不用再划分
            leaf_vectors = vector_data[indices]
            self.mean_vector = np.mean(leaf_vectors, axis=0)
            self.max_radius = np.max(
                np.linalg.norm(leaf_vectors - self.mean_vector, axis=1)
            )
        else:
            # 非叶子节点继续划分
            sorted_idx = indices[np.argsort(scalar_data[indices][:, self.axis])]
            median = len(sorted_idx) // 2
            self.split = scalar_data[sorted_idx[median]][self.axis]

            self.left = KDNode(sorted_idx[:median], depth + 1)
            self.right = KDNode(sorted_idx[median:], depth + 1)

            # 内部节点存下层向量代表信息
            all_vectors = vector_data[indices]
            self.mean_vector = np.mean(all_vectors, axis=0)
            self.max_radius = np.max(
                np.linalg.norm(all_vectors - self.mean_vector, axis=1)
            )

            # 小规模 KMeans 聚类中心（可选，probe时可用）
            if len(indices) >= ivf_clusters:
                kmeans = KMeans(
                    n_clusters=min(ivf_clusters, len(indices)),
                    random_state=42,
                    n_init=3,
                    max_iter=50,
                )
                kmeans.fit(all_vectors)
                self.node_centroids = kmeans.cluster_centers_


print("Building KDTree...")
root = KDNode(np.arange(N))
print("KDTree built.")

# ==============================
# 3. 收集叶子并建 IVF
# ==============================
leaf_nodes = []


def collect_leaves(node):
    if node.is_leaf:
        leaf_nodes.append(node)
    else:
        collect_leaves(node.left)
        collect_leaves(node.right)


collect_leaves(root)

ivf_index = {}
print("Building IVF in leaves...")

for lid, node in enumerate(leaf_nodes):
    indices = node.indices
    vectors = vector_data[indices]

    if len(indices) < ivf_clusters:
        ivf_index[lid] = {
            "centroids": None,
            "labels": None,
            "indices": indices,
            "vectors": vectors,
        }
        continue

    kmeans = KMeans(
        n_clusters=ivf_clusters,
        random_state=42,
        n_init=5,
        max_iter=100,
    )
    kmeans.fit(vectors)

    ivf_index[lid] = {
        "centroids": kmeans.cluster_centers_,
        "labels": kmeans.labels_,
        "indices": indices,
        "vectors": vectors,
    }

print("IVF built.")


# ==============================
# 工具函数
# ==============================
def scalar_match(row):
    for d in range(scalar_dim):
        if row[d] < query_ranges[d][0] or row[d] > query_ranges[d][1]:
            return False
    return True


def bbox_intersect(node):
    for d in range(scalar_dim):
        if (
            node.bbox_max[d] < query_ranges[d][0]
            or node.bbox_min[d] > query_ranges[d][1]
        ):
            return False
    return True


def prune_by_node(node, query_vector, heap):
    """利用内部节点均值+半径粗剪"""
    if len(heap) < topk:
        return False  # topk 未满，不剪
    dist_to_mean = np.linalg.norm(query_vector - node.mean_vector)
    if dist_to_mean - node.max_radius > -heap[0][0]:
        return True  # 剪枝
    return False


# ==============================
# 4. 查询 (加入内部节点剪枝)
# ==============================
def hybrid_query(query_vector):
    heap = []

    def recurse(node):
        if not bbox_intersect(node):
            return
        if prune_by_node(node, query_vector, heap):
            return

        if node.is_leaf:
            lid = leaf_nodes.index(node)
            leaf_info = ivf_index[lid]
            indices = leaf_info["indices"]
            vectors = leaf_info["vectors"]
            centroids = leaf_info["centroids"]
            labels = leaf_info["labels"]

            if centroids is None:
                for i, idx in enumerate(indices):
                    if not scalar_match(scalar_data[idx]):
                        continue
                    dist = np.linalg.norm(vectors[i] - query_vector)
                    if len(heap) < topk:
                        heapq.heappush(heap, (-dist, idx))
                    elif dist < -heap[0][0]:
                        heapq.heapreplace(heap, (-dist, idx))
                return

            centroid_dists = np.linalg.norm(centroids - query_vector, axis=1)
            nearest = np.argsort(centroid_dists)[:probe_clusters]

            for cid in nearest:
                mask = labels == cid
                local_ids = np.where(mask)[0]
                for lid2 in local_ids:
                    idx = indices[lid2]
                    if not scalar_match(scalar_data[idx]):
                        continue
                    dist = np.linalg.norm(vectors[lid2] - query_vector)
                    if len(heap) < topk:
                        heapq.heappush(heap, (-dist, idx))
                    elif dist < -heap[0][0]:
                        heapq.heapreplace(heap, (-dist, idx))
            return

        recurse(node.left)
        recurse(node.right)

    recurse(root)
    return sorted([idx for _, idx in heap])


# ==============================
# 5. Ground Truth
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

print("Ground truth ready.")

# ==============================
# 6. 计算 Recall
# ==============================
print("Evaluating recall...")
total_recall = 0

for qi, q in enumerate(query_vectors):
    approx = hybrid_query(q)
    gt = set(ground_truth[qi])
    hit = len(set(approx) & gt)
    total_recall += hit / topk

recall = total_recall / Q
print(f"\nRecall@{topk}: {recall:.4f}")
