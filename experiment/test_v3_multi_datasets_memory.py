import random
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc
import gc
from sklearn.cluster import KMeans

# 设置随机种子
np.random.seed(42)
random.seed(42)

# ==========================================================
# 参数配置
# ==========================================================
VECTOR_DIM = 32
LEAF_SIZE = 200  # 对齐你原始索引的参数
IVF_CLUSTERS = 8  # 对齐你原始索引的参数
DATA_SIZES = [10000, 30000, 50000, 100000]


# ==========================================================
# 数据生成
# ==========================================================
def generate_data(n):
    vectors = np.random.randn(n, VECTOR_DIM).astype(np.float32)  # 对齐原始数据类型
    scalars = np.random.rand(n, 5).astype(np.float32)
    return vectors, scalars


# ==========================================================
# 1️⃣ KD-Tree (标量索引)
# ==========================================================
class KDNode:
    __slots__ = [
        "left",
        "right",
        "is_leaf",
        "indices",
        "axis",
        "split",
        "bbox_min",
        "bbox_max",
        "mean_vector",
        "max_radius",
    ]

    def __init__(self, indices, depth, scalars, vectors):
        self.left = None
        self.right = None
        self.is_leaf = len(indices) <= LEAF_SIZE

        # 标量包围盒 (保留完整结构以准确测量内存)
        self.bbox_min = np.min(scalars[indices], axis=0)
        self.bbox_max = np.max(scalars[indices], axis=0)

        # 向量统计 (保留完整结构)
        vecs = vectors[indices]
        self.mean_vector = np.mean(vecs, axis=0)
        self.max_radius = np.max(np.linalg.norm(vecs - self.mean_vector, axis=1))

        if self.is_leaf:
            self.indices = indices
            self.axis = self.split = None
            return

        # 非叶子节点分割
        self.axis = depth % scalars.shape[1]
        sorted_idx = indices[np.argsort(scalars[indices][:, self.axis])]
        mid = len(sorted_idx) // 2
        self.split = scalars[sorted_idx[mid]][self.axis]
        self.indices = None  # 非叶子不存索引省内存

        self.left = KDNode(sorted_idx[:mid], depth + 1, scalars, vectors)
        self.right = KDNode(sorted_idx[mid:], depth + 1, scalars, vectors)


def build_kdtree(scalars, vectors):
    indices = np.arange(len(scalars))
    return KDNode(indices, 0, scalars, vectors)


# ==========================================================
# 2️⃣ IVF (纯向量索引)
# ==========================================================
def build_ivf(vectors):
    if len(vectors) < IVF_CLUSTERS:
        return {"centroids": None, "labels": None, "vectors": vectors}
    kmeans = KMeans(
        n_clusters=IVF_CLUSTERS,
        n_init=1,
        max_iter=30,
        random_state=42,
        algorithm="lloyd",
    )
    kmeans.fit(vectors)
    return {
        "centroids": kmeans.cluster_centers_,
        "labels": kmeans.labels_,
        "vectors": vectors,
    }


# ==========================================================
# 3️⃣ 简化 HNSW (纯向量索引)
# ==========================================================
class SimpleHNSW:
    __slots__ = ["M", "nodes", "graph"]

    def __init__(self, M=16):  # 对齐hnswlib默认参数
        self.M = M
        self.nodes = []
        self.graph = {}

    def build(self, vectors):
        for i, v in enumerate(vectors):
            current_len = len(self.nodes)
            if current_len > 0:
                neighbors = random.sample(
                    range(current_len),
                    min(self.M, current_len),
                )
            else:
                neighbors = []
            self.nodes.append(v)
            self.graph[i] = neighbors


# ==========================================================
# 4️⃣ KD-ANN (你的混合索引: KD-Tree + IVF)
# ==========================================================
class KDANNNode:
    __slots__ = [
        "left",
        "right",
        "is_leaf",
        "indices",
        "axis",
        "split",
        "bbox_min",
        "bbox_max",
        "mean_vector",
        "max_radius",
        "ivf_centroids",
        "ivf_labels",
        "ivf_vectors",
    ]

    def __init__(self, indices, depth, scalars, vectors):
        self.left = None
        self.right = None
        self.is_leaf = len(indices) <= LEAF_SIZE

        # 标量包围盒
        self.bbox_min = np.min(scalars[indices], axis=0)
        self.bbox_max = np.max(scalars[indices], axis=0)

        # 向量统计
        vecs = vectors[indices]
        self.mean_vector = np.mean(vecs, axis=0)
        self.max_radius = np.max(np.linalg.norm(vecs - self.mean_vector, axis=1))

        if self.is_leaf:
            self.indices = indices
            self.axis = self.split = None
            self._build_leaf_ivf(vectors[indices])
            return

        # 非叶子节点分割
        self.axis = depth % scalars.shape[1]
        sorted_idx = indices[np.argsort(scalars[indices][:, self.axis])]
        mid = len(sorted_idx) // 2
        self.split = scalars[sorted_idx[mid]][self.axis]
        self.indices = None
        self.ivf_centroids = self.ivf_labels = self.ivf_vectors = None

        self.left = KDANNNode(sorted_idx[:mid], depth + 1, scalars, vectors)
        self.right = KDANNNode(sorted_idx[mid:], depth + 1, scalars, vectors)

    def _build_leaf_ivf(self, leaf_vectors):
        n = len(leaf_vectors)
        if n < IVF_CLUSTERS:
            self.ivf_centroids = self.ivf_labels = None
        else:
            kmeans = KMeans(
                n_clusters=min(IVF_CLUSTERS, n), n_init=1, max_iter=30, random_state=42
            )
            kmeans.fit(leaf_vectors)
            self.ivf_centroids = kmeans.cluster_centers_
            self.ivf_labels = kmeans.labels_
        self.ivf_vectors = leaf_vectors


class KDANNIndex:
    def __init__(self, vectors, scalars):
        self.vectors = vectors
        self.scalars = scalars
        self.root = KDANNNode(np.arange(len(vectors)), 0, scalars, vectors)


def build_kd_ann(vectors, scalars):
    return KDANNIndex(vectors, scalars)


# ==========================================================
# β-WST (Simplified)
# ==========================================================
BETA = 4


class BetaWSTNode:
    __slots__ = [
        "children",
        "is_leaf",
        "indices",
        "bbox_min",
        "bbox_max",
        "mean_vector",
        "max_radius",
        "ivf_centroids",
        "ivf_labels",
        "ivf_vectors",
    ]

    def __init__(self, indices, depth, scalars, vectors):
        self.children = []

        self.is_leaf = len(indices) <= LEAF_SIZE

        # scalar bbox
        self.bbox_min = np.min(scalars[indices], axis=0)
        self.bbox_max = np.max(scalars[indices], axis=0)

        # vector stats
        vecs = vectors[indices]
        self.mean_vector = np.mean(vecs, axis=0)
        self.max_radius = np.max(np.linalg.norm(vecs - self.mean_vector, axis=1))

        if self.is_leaf:
            self.indices = indices
            self._build_leaf_ivf(vectors[indices])
            return

        self.indices = None

        # 当前划分维度
        axis = depth % scalars.shape[1]

        # 排序
        sorted_idx = indices[np.argsort(scalars[indices][:, axis])]

        # β-way split
        chunks = np.array_split(sorted_idx, BETA)

        for chunk in chunks:
            if len(chunk) == 0:
                continue

            child = BetaWSTNode(
                chunk,
                depth + 1,
                scalars,
                vectors,
            )

            self.children.append(child)

    def _build_leaf_ivf(self, leaf_vectors):
        n = len(leaf_vectors)

        if n < IVF_CLUSTERS:
            self.ivf_centroids = None
            self.ivf_labels = None
        else:
            kmeans = KMeans(
                n_clusters=min(IVF_CLUSTERS, n),
                n_init=1,
                max_iter=30,
                random_state=42,
            )

            kmeans.fit(leaf_vectors)

            self.ivf_centroids = kmeans.cluster_centers_
            self.ivf_labels = kmeans.labels_

        self.ivf_vectors = leaf_vectors


class BetaWSTIndex:
    def __init__(self, vectors, scalars):
        self.vectors = vectors
        self.scalars = scalars

        self.root = BetaWSTNode(
            np.arange(len(vectors)),
            0,
            scalars,
            vectors,
        )


def build_beta_wst(vectors, scalars):
    return BetaWSTIndex(vectors, scalars)


# ==========================================================
# 内存测量函数 (严格清理)
# ==========================================================
def measure_memory(func, *args):
    gc.collect()
    gc.disable()  # 禁用GC防止干扰
    tracemalloc.start()

    try:
        result = func(*args)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # 立即删除结果并强制回收
        del result
        gc.collect()
        gc.enable()

        return peak / 1024 / 1024  # 转换为MB
    except Exception as e:
        tracemalloc.stop()
        tracemalloc.clear_traces()
        gc.enable()
        gc.collect()
        raise e


# ==========================================================
# 实验主循环
# ==========================================================
results = []

# 定义要测试的方法列表
# 顺序: KD-Tree, IVF, HNSW, KD-ANN
# method_names = ["KD-Tree", "IVF", "HNSW", "KD-ANN"]
method_names = ["KD-Tree", "β-WST"]

print(f"{'Size':<10} | " + " | ".join(f"{name:<10}" for name in method_names))
print("-" * (10 + 3 + 14 * len(method_names)))

for N in DATA_SIZES:
    print(f"\nTesting size {N}...")
    vectors, scalars = generate_data(N)

    # row = []

    # # 1. KD-Tree
    # mem = measure_memory(build_kdtree, scalars, vectors)
    # row.append(mem)
    # print(f"  KD-Tree: {mem:.2f} MB")

    # # 2. IVF
    # mem = measure_memory(build_ivf, vectors)
    # row.append(mem)
    # print(f"  IVF: {mem:.2f} MB")

    # # 3. HNSW
    # def run_hnsw(vecs):
    #     h = SimpleHNSW()
    #     h.build(vecs)
    #     return h

    # mem = measure_memory(run_hnsw, vectors)
    # row.append(mem)
    # print(f"  HNSW: {mem:.2f} MB")

    # # 4. KD-ANN
    # mem = measure_memory(build_kd_ann, vectors, scalars)
    # row.append(mem)
    # print(f"  KD-ANN: {mem:.2f} MB")

    # results.append(row)
    # print(f"{N:<10} | " + " | ".join(f"{val:<10.2f}" for val in row))

    row = []

# KD-ANN
    mem = measure_memory(build_kd_ann, vectors, scalars)
    row.append(mem)
    print(f"  KD-ANN: {mem:.2f} MB")

    # β-WST
    mem = measure_memory(build_beta_wst, vectors, scalars)
    row.append(mem)
    print(f"  β-WST: {mem:.2f} MB")


# ==========================================================
# 优化版画图（论文风 + 清晰对比）
# ==========================================================

colors = ["#4C72B0", "#55A868", "#8172B3", "#C44E52"]  # 蓝、绿、紫、红
hatches = ["//", "\\\\", "--", "xx"]

num_sizes = len(DATA_SIZES)
num_methods = len(method_names)

fig, axes = plt.subplots(1, num_sizes, figsize=(4 * num_sizes, 4), sharey=True)

if num_sizes == 1:
    axes = [axes]

bar_width = 0.15

for i, ax in enumerate(axes):
    mem_usage = results[i]
    center = 0
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, num_methods)

    for j in range(num_methods):
        ax.bar(
            center + offsets[j],
            mem_usage[j],
            width=bar_width,
            color=colors[j],
            hatch=hatches[j],
            edgecolor="black",
            linewidth=0.6,
        )
        # 添加数值标签
        ax.text(
            center + offsets[j],
            mem_usage[j] * 1.02,
            f"{mem_usage[j]:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(f"{DATA_SIZES[i]//1000}K Data", fontsize=11)
    ax.set_xticks([])
    ax.set_yscale("log")  # 对数轴更易看差异
    ax.grid(axis="y", linestyle="--", alpha=0.3)

# 只在第一个子图显示y轴
axes[0].set_ylabel("Peak Memory (MB)", fontsize=11)

# 图例
from matplotlib.patches import Patch

legend_elements = [
    Patch(
        facecolor=colors[j], edgecolor="black", hatch=hatches[j], label=method_names[j]
    )
    for j in range(num_methods)
]
axes[0].legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=9)

plt.subplots_adjust(wspace=0.2)
plt.tight_layout()
plt.savefig("memory_comparison.png", dpi=300, bbox_inches="tight")
print("\n✅ Plot saved to 'memory_comparison.png'")
plt.show()
