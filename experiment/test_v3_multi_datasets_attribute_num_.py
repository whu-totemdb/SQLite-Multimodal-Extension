import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.interpolate import PchipInterpolator
import mytool

# ==============================
# 固定最佳索引参数
# ==============================

leaf_size = 512
ivf_clusters = 128
probe_clusters = 8

topk = 10
attribute_dims = list(range(1, 17))
selectivities = [0.05, 0.1, 0.3]

colors = ["tab:blue", "tab:orange", "tab:green"]
markers = ["o", "s", "^"]

# ==============================
# 数据集
# ==============================

dataset = {"name": "RF1M", "type": "npy", "path": "random_ints_1m"}

print(f"Running dataset: {dataset['name']}")
vector_data, query_vectors = mytool.load_datasets(dataset["type"], dataset["path"])

vector_data = vector_data[:30000]
query_vectors = query_vectors[:500]

N = vector_data.shape[0]
Q = query_vectors.shape[0]

# ==============================
# 构建 KD-tree + IVF
# ==============================


def build_index(vector_data, scalar_data):

    scalar_dim = scalar_data.shape[1]

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

    return root, len(leaf_nodes)


# ==============================
# 主实验
# ==============================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for s_idx, selectivity in enumerate(selectivities):

    latency_curve = []
    qps_curve = []

    for dim in attribute_dims:

        scalar_dim = dim
        np.random.seed(42)
        scalar_data = np.random.rand(N, scalar_dim)

        # 固定 selectivity
        width = selectivity ** (1.0 / scalar_dim)
        query_ranges = [(0.5 - width / 2, 0.5 + width / 2)] * scalar_dim

        def bbox_intersect(node):
            for d in range(scalar_dim):
                if (
                    node.bbox_max[d] < query_ranges[d][0]
                    or node.bbox_min[d] > query_ranges[d][1]
                ):
                    return False
            return True

        root, total_leaf_cnt = build_index(vector_data, scalar_data)

        total_visited = 0

        start = time.perf_counter()

        for _ in range(Q):

            def recurse(node):
                if not bbox_intersect(node):
                    return 0
                if node.is_leaf:
                    return 1
                return recurse(node.left) + recurse(node.right)

            total_visited += recurse(root)

        end = time.perf_counter()

        total_time = end - start
        avg_latency = (total_time / Q) * 1000  # 转为 ms
        qps = Q / total_time

        latency_curve.append(avg_latency)
        qps_curve.append(qps)

        print(f"Selectivity {selectivity}, Dim {dim} done")

    x = np.array(attribute_dims)

    latency_smooth = PchipInterpolator(x, latency_curve)(x)
    qps_smooth = PchipInterpolator(x, qps_curve)(x)

    # Latency 图
    axes[0].scatter(
        x,
        latency_curve,
        marker=markers[s_idx],
        color=colors[s_idx],
        s=45,
        zorder=3,
    )
    axes[0].plot(
        x,
        latency_smooth,
        color=colors[s_idx],
        linewidth=2,
        label=f"Sel={selectivity}",
    )

    # QPS 图（保持原逻辑）
    axes[1].scatter(
        x,
        qps_curve,
        marker=markers[s_idx],
        color=colors[s_idx],
        s=45,
        zorder=3,
    )
    axes[1].plot(
        x,
        qps_smooth,
        color=colors[s_idx],
        linewidth=2,
        label=f"Sel={selectivity}",
    )

# ==============================
# 图形美化
# ==============================

# Latency
axes[0].set_title("Average Query Latency")
axes[0].set_xlabel("Attribute Dimension")
axes[0].set_ylabel("Latency (ms)")
axes[0].grid(True, linestyle="--", alpha=0.35)

# QPS
axes[1].set_title("Throughput (QPS)")
axes[1].set_xlabel("Attribute Dimension")
axes[1].set_ylabel("Queries Per Second")
axes[1].set_yscale("log")
axes[1].grid(True, linestyle="--", alpha=0.35)

for ax in axes:
    ax.legend(frameon=False)

plt.tight_layout()
plt.show()
