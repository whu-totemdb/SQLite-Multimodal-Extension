import numpy as np
import heapq
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.interpolate import PchipInterpolator
import mytool

# ==============================
# 全局参数
# ==============================

scalar_dim = 5
topk = 10
leaf_size = 512
# probe_range = range(1, 17)
ivf_clusters_list = [32, 64, 128, 256]
query_ranges = [(0.35, 0.65)] * scalar_dim

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
markers = ["o", "s", "^", "D"]


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


# ==============================
# 构建索引
# ==============================


def build_index(vector_data, scalar_data, ivf_clusters):

    N = vector_data.shape[0]

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
# 查询
# ==============================


def hybrid_query(
    query_vector, probe_clusters, root, ivf_index, vector_data, scalar_data
):

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
# 单数据集实验
# ==============================


def run_experiment(vector_data, query_vectors):

    vector_data = vector_data[:30000]
    query_vectors = query_vectors[:200]

    N = vector_data.shape[0]
    Q = query_vectors.shape[0]

    np.random.seed(42)
    scalar_data = np.random.rand(N, scalar_dim)

    # 生成 GT
    ground_truth = []
    for q in query_vectors:
        dists = np.linalg.norm(vector_data - q, axis=1)
        valid_mask = np.array([scalar_match(scalar_data[i]) for i in range(N)])
        filtered_idx = np.where(valid_mask)[0]
        filtered_dists = dists[filtered_idx]
        top_ids = filtered_idx[np.argsort(filtered_dists)[:topk]]
        ground_truth.append(set(top_ids))

    group_results = []

    for ivf_clusters in ivf_clusters_list:

        root, ivf_index = build_index(vector_data, scalar_data, ivf_clusters)

        recalls = []
        qpss = []
        probe_range = range(1, int(ivf_clusters / 2) + 1)
        for probe in probe_range:

            total_recall = 0
            start = time.perf_counter()

            for qi in range(Q):
                approx = hybrid_query(
                    query_vectors[qi],
                    probe,
                    root,
                    ivf_index,
                    vector_data,
                    scalar_data,
                )
                hit = len(set(approx) & ground_truth[qi])
                total_recall += hit / topk

            end = time.perf_counter()

            recall = total_recall / Q
            qps = Q / (end - start)

            recalls.append(recall)
            qpss.append(qps)

        group_results.append((recalls, qpss))

    return group_results


# ==============================
# 多数据集主程序
# ==============================

datasets = [
    {"name": "SIFT", "type": "fvecs", "path": "sift10k"},
    {"name": "hub-medium", "type": "npy", "path": "random_ints_1m"},
    {"name": "laion", "type": "npy", "path": "laion-small-clip"},
    {"name": "rf1m", "type": "npy", "path": "random_float_1m"},
]

# ==============================
# 2x2 多数据集绘图
# ==============================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(11, 8),
    sharex=False,
    sharey=False,
)

axes = axes.flatten()

legend_handles = []
legend_labels = []

for i, ds in enumerate(datasets):

    print(f"\nRunning dataset: {ds['name']}")

    vector_data, query_vectors = mytool.load_datasets(ds["type"], ds["path"])
    group_results = run_experiment(vector_data, query_vectors)

    ax = axes[i]

    all_recalls = []
    all_qps = []

    for idx, (recalls, qpss) in enumerate(group_results):

        sorted_pairs = sorted(zip(recalls, qpss))
        recalls_sorted = np.array([p[0] for p in sorted_pairs])
        qps_sorted = np.array([p[1] for p in sorted_pairs])

        unique_recalls, unique_indices = np.unique(recalls_sorted, return_index=True)
        unique_qps = qps_sorted[unique_indices]

        all_recalls.extend(unique_recalls)
        all_qps.extend(unique_qps)

        sc = ax.scatter(
            unique_recalls,
            unique_qps,
            color=colors[idx],
            marker=markers[idx],
            s=22,
            alpha=0.85,
            zorder=3,
        )

        if len(unique_recalls) >= 3:
            interp = PchipInterpolator(unique_recalls, unique_qps)
            x_new = np.linspace(
                unique_recalls.min(),
                unique_recalls.max(),
                200,
            )
            y_new = interp(x_new)

            ax.plot(
                x_new,
                y_new,
                color=colors[idx],
                linewidth=1.5,
                zorder=2,
            )

        # 只收集一次图例
        if i == 0:
            legend_handles.append(sc)
            legend_labels.append(f"IVF = {ivf_clusters_list[idx]}")

    # ==========================
    # 🔥 自适应缩放（贴边）
    # ==========================

    xmin, xmax = min(all_recalls), max(all_recalls)
    ymin, ymax = min(all_qps), max(all_qps)

    x_margin = (xmax - xmin) * 0.05
    ax.set_xlim(xmin - x_margin, xmax + x_margin)

    ax.set_ylim(ymin * 0.9, ymax * 1.1)

    ax.set_title(ds["name"], fontsize=11)
    ax.set_xlabel("Recall@10")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)

    if i % 2 == 0:
        ax.set_ylabel("QPS (1/s)")

# ==============================
# 🔥 顶部统一图例
# ==============================

fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.97),
    ncol=4,
    frameon=False,
    fontsize=10,
)

# ==============================
# 🔥 控制整体布局
# ==============================

fig.subplots_adjust(
    top=0.90,  # 给图例留空间
    hspace=0.28,
    wspace=0.22,
)

plt.show()
