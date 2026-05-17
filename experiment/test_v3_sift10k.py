import numpy as np
import heapq
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.interpolate import PchipInterpolator
from huggingface_hub import hf_hub_download
import mytool

# ==============================
# 1. 读取 fvecs
# ==============================


print("Loading dataset...")

dataset_type = "npy"  # 改成 "fvecs" 或 "npy"

vector_data, query_vectors = mytool.load_dataset(dataset_type)

# 控制规模（避免太慢）
vector_data = vector_data[:10000]
query_vectors = query_vectors[:100]

N, dim = vector_data.shape
Q = query_vectors.shape[0]

print(f"N={N}, Q={Q}, dim={dim}")

# ==============================
# 2. 随机 scalar 数据
# ==============================

scalar_dim = 5
np.random.seed(42)
scalar_data = np.random.rand(N, scalar_dim)

# 收紧过滤范围（增加区分度）
query_ranges = [(0.35, 0.65)] * scalar_dim

topk = 10
leaf_size = 256
ivf_clusters_list = [16, 32, 64, 128]


# ==============================
# 3. 构建索引函数
# ==============================


def build_index(ivf_clusters):

    class KDNode:
        def __init__(self, indices, depth=0):
            self.indices = indices
            self.left = None
            self.right = None
            self.axis = depth % scalar_dim
            self.leaf_id = None

            self.bbox_min = np.min(scalar_data[indices], axis=0)
            self.bbox_max = np.max(scalar_data[indices], axis=0)

            self.is_leaf = len(indices) <= leaf_size

            if self.is_leaf:
                return

            sorted_idx = indices[np.argsort(scalar_data[indices][:, self.axis])]
            median = len(sorted_idx) // 2

            self.left = KDNode(sorted_idx[:median], depth + 1)
            self.right = KDNode(sorted_idx[median:], depth + 1)

    root = KDNode(np.arange(N))

    leaf_nodes = []

    def collect(node):
        if node.is_leaf:
            node.leaf_id = len(leaf_nodes)
            leaf_nodes.append(node)
        else:
            collect(node.left)
            collect(node.right)

    collect(root)

    ivf_index = {}

    print(f"Building IVF (clusters={ivf_clusters})")

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
            n_init=3,
            max_iter=50,
        )
        kmeans.fit(vectors)

        ivf_index[lid] = {
            "centroids": kmeans.cluster_centers_,
            "labels": kmeans.labels_,
            "indices": indices,
            "vectors": vectors,
        }

    return root, ivf_index


# ==============================
# 4. 工具函数
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


def hybrid_query(query_vector, probe_clusters, root, ivf_index):

    heap = []

    def recurse(node):
        if not bbox_intersect(node):
            return

        if node.is_leaf:
            leaf_info = ivf_index[node.leaf_id]
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
                local_ids = np.where(labels == cid)[0]
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
    return [idx for _, idx in heap]


# ==============================
# 5. Ground Truth
# ==============================

print("Computing Ground Truth...")

ground_truth = []

for q in query_vectors:
    dists = np.linalg.norm(vector_data - q, axis=1)
    valid_mask = np.array([scalar_match(scalar_data[i]) for i in range(N)])
    filtered_idx = np.where(valid_mask)[0]
    filtered_dists = dists[filtered_idx]
    top_ids = filtered_idx[np.argsort(filtered_dists)[:topk]]
    ground_truth.append(set(top_ids))

print("GT ready.")


# ==============================
# 6. 评测函数
# ==============================


def evaluate_once(probe, root, ivf_index):

    total_recall = 0
    start = time.perf_counter()

    for qi in range(Q):
        approx = hybrid_query(query_vectors[qi], probe, root, ivf_index)
        hit = len(set(approx) & ground_truth[qi])
        total_recall += hit / topk

    end = time.perf_counter()

    recall = total_recall / Q
    qps = Q / (end - start)

    return recall, qps


# ==============================
# 7. 主实验
# ==============================

probe_range = range(1, 17)
group_results = []
all_points = []

for ivf_clusters in ivf_clusters_list:

    root, ivf_index = build_index(ivf_clusters)

    recalls = []
    qpss = []

    print(f"\nRunning ivf_clusters={ivf_clusters}")

    for probe in probe_range:
        r, q = evaluate_once(probe, root, ivf_index)
        recalls.append(r)
        qpss.append(q)
        all_points.append((r, q))
        print(f"  probe={probe}: recall={r:.4f}, qps={q:.2f}")

    group_results.append((recalls, qpss))


# ==============================
# 8. 画图（四条线分别拟合 + 不同点形状）
# ==============================

plt.figure(figsize=(8, 6))

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
markers = ["o", "s", "^", "D"]  # 🔑 新增：圆形、方形、三角、菱形

for idx, (recalls, qpss) in enumerate(group_results):

    # 按 recall 排序（从左到右）
    sorted_pairs = sorted(zip(recalls, qpss))
    recalls_sorted = np.array([p[0] for p in sorted_pairs])
    qps_sorted = np.array([p[1] for p in sorted_pairs])

    # 去重（避免插值报错）
    unique_recalls, unique_indices = np.unique(recalls_sorted, return_index=True)
    unique_qps = qps_sorted[unique_indices]

    # 🔑 原始点：添加 marker 参数
    plt.scatter(
        unique_recalls,
        unique_qps,
        color=colors[idx],
        marker=markers[idx],  # 🔑 每组不同形状
        s=30,  # 点大小（面积）
        edgecolors="white",  # 白边更清晰
        linewidth=0.5,
        alpha=0.8,
        label=f"IVF = {ivf_clusters_list[idx]}",  # 🔑 散点也加标签
        zorder=3,
    )

    # 只有点数>=3才拟合
    if len(unique_recalls) >= 3:
        interp = PchipInterpolator(unique_recalls, unique_qps)

        x_new = np.linspace(unique_recalls.min(), unique_recalls.max(), 200)

        y_new = interp(x_new)

        plt.plot(
            x_new,
            y_new,
            color=colors[idx],
            linewidth=1.5,  # 曲线稍细一点
            label=f"IVF clusters = {ivf_clusters_list[idx]}",
            zorder=2,
        )

plt.xlabel("Recall@10", fontsize=11)
plt.ylabel("QPS (1/s)", fontsize=11)
plt.title("Recall-QPS Tradeoff", fontsize=12, pad=15)
plt.yscale("log")

# 🔑 图例：横排显示，放顶部
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=4,
    frameon=True,
    framealpha=0.9,
    fontsize=9,
    columnspacing=1.5,
    handlelength=1.5,
)

plt.grid(True, linestyle="--", alpha=0.5, which="both", axis="y")
plt.tight_layout()
plt.show()
