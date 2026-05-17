import numpy as np
import time
import random
import json
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import hnswlib


plt.rcParams["font.sans-serif"] = ["SimHei"]  # 仅用于中文标题
plt.rcParams["axes.unicode_minus"] = False

# ==========================================================
# 参数配置 (对齐你的原始索引参数)
# ==========================================================

SCALAR_DIM = 5
VECTOR_DIM = 64  # 若使用真实数据可自动获取，这里为生成数据预设
LEAF_SIZE = 200  # 对齐你的 leaf_size=200
IVF_CLUSTERS = 8  # 对齐你的 ivf_clusters=8
PROBE_CLUSTERS = 4  # 对齐你的 probe_clusters=4

DATA_SIZES = [10000, 30000, 50000, 100000]  # 可按需调整
Q = 100  # 查询数量（当前仅测构建时间，暂未用到）
topk = 10

np.random.seed(42)
random.seed(42)

# ==========================================================
# 读取 fvecs (保留接口，若用生成数据可忽略)
# ==========================================================


def read_fvecs(filename):
    vectors = []
    with open(filename, "rb") as f:
        while True:
            dim = np.fromfile(f, dtype=np.int32, count=1)
            if not dim.size:
                break
            vec = np.fromfile(f, dtype=np.float32, count=dim[0])
            vectors.append(vec)
    return np.vstack(vectors)


# ==========================================================
# 数据加载 (支持真实数据 + 生成数据 fallback)
# ==========================================================


def load_datasets(
    dataset_type="generate", base_path="./random_ints_1m", n=100000, vector_dim=64
):
    if dataset_type == "fvecs":
        vector_data = read_fvecs(f"{base_path}/vectors.fvecs")
        query_vectors = read_fvecs(f"{base_path}/query.fvecs")
    elif dataset_type == "npy":
        vector_data = np.load(f"{base_path}/vectors.npy")
        query_vectors = []
        with open(f"{base_path}/tests.jsonl", "r") as f:
            for line in f:
                obj = json.loads(line)
                query_vectors.append(obj["query"])
        query_vectors = np.array(query_vectors, dtype=np.float32)
    elif dataset_type == "generate":
        # 生成模拟数据（对齐你的原始数据生成逻辑）
        vector_data = np.random.randn(n, vector_dim).astype(np.float32)
        query_vectors = np.random.randn(Q, vector_dim).astype(np.float32)
    else:
        raise ValueError("Unsupported dataset type")
    return vector_data.astype(np.float32), query_vectors.astype(np.float32)


# ==========================================================
# 生成 scalar 属性
# ==========================================================


def generate_scalars(n):
    return np.random.rand(n, SCALAR_DIM).astype(np.float32)


# ==========================================================
# 1. KD-Tree (sklearn, 标量)
# ==========================================================


def build_kdtree(scalars):
    return KDTree(scalars, leaf_size=LEAF_SIZE)


# ==========================================================
# 2. IVF (普通, 向量)
# ==========================================================


def build_ivf(vectors):
    n_clusters = min(IVF_CLUSTERS, len(vectors))
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100, random_state=42)
    kmeans.fit(vectors)
    return {
        "centroids": kmeans.cluster_centers_,
        "labels": kmeans.labels_,
        "vectors": vectors,
    }


# ==========================================================
# 3. HNSW (hnswlib, 向量)
# ==========================================================


def build_hnsw(vectors):
    dim = vectors.shape[1]
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
    index.add_items(vectors)
    return index


# ==========================================================
# 4.  KDIVF (你的混合索引: 标量KD + 向量IVF)
# ==========================================================


class KDIVFNode:
    """KDIVF 索引的树节点"""

    def __init__(self, indices, scalars, vectors, depth=0):
        self.indices = indices
        self.is_leaf = len(indices) <= LEAF_SIZE

        # 标量包围盒 (用于范围剪枝)
        self.bbox_min = np.min(scalars[indices], axis=0)
        self.bbox_max = np.max(scalars[indices], axis=0)

        # 向量统计 (用于路由+剪枝)
        vecs = vectors[indices]
        self.mean_vector = np.mean(vecs, axis=0)
        self.max_radius = np.max(np.linalg.norm(vecs - self.mean_vector, axis=1))

        if self.is_leaf:
            # 叶子: 构建局部IVF
            self.axis = self.split = None
            self.left = self.right = None
            self._build_leaf_ivf(vectors[indices])
        else:
            # 非叶子: 按标量维度分割
            self.axis = depth % scalars.shape[1]
            sorted_idx = indices[np.argsort(scalars[indices][:, self.axis])]
            mid = len(sorted_idx) // 2
            self.split = scalars[sorted_idx[mid]][self.axis]
            self.left = KDIVFNode(sorted_idx[:mid], scalars, vectors, depth + 1)
            self.right = KDIVFNode(sorted_idx[mid:], scalars, vectors, depth + 1)
            # 叶子IVF相关属性置空
            self.ivf_centroids = self.ivf_labels = self.ivf_vectors = None

    def _build_leaf_ivf(self, leaf_vectors):
        """构建叶子节点的局部IVF索引"""
        n = len(leaf_vectors)
        if n < IVF_CLUSTERS:
            self.ivf_centroids = self.ivf_labels = None
        else:
            kmeans = KMeans(
                n_clusters=min(IVF_CLUSTERS, n), n_init=5, max_iter=50, random_state=42
            )
            kmeans.fit(leaf_vectors)
            self.ivf_centroids = kmeans.cluster_centers_
            self.ivf_labels = kmeans.labels_
        self.ivf_vectors = leaf_vectors


class KDIVFIndex:
    """KDIVF 混合索引主类 (KD-Tree + IVF)"""

    def __init__(self, vectors, scalars):
        self.vectors = vectors
        self.scalars = scalars
        self.scalar_dim = scalars.shape[1]
        self.root = KDIVFNode(np.arange(len(vectors)), scalars, vectors)

    def query(self, query_vector, query_ranges, topk=10, probe_clusters=PROBE_CLUSTERS):
        """混合查询: 标量范围 + 向量近邻"""
        import heapq

        heap = []  # 最大堆: (-dist, global_idx)

        def scalar_match(row):
            return all(
                query_ranges[d][0] <= row[d] <= query_ranges[d][1]
                for d in range(self.scalar_dim)
            )

        def bbox_intersect(node):
            for d in range(self.scalar_dim):
                if (
                    node.bbox_max[d] < query_ranges[d][0]
                    or node.bbox_min[d] > query_ranges[d][1]
                ):
                    return False
            return True

        def prune_by_node(node):
            if len(heap) < topk:
                return False
            dist_to_mean = np.linalg.norm(query_vector - node.mean_vector)
            return dist_to_mean - node.max_radius > -heap[0][0]

        def update_heap(dist, idx):
            if len(heap) < topk:
                heapq.heappush(heap, (-dist, idx))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, idx))

        def recurse(node):
            if not bbox_intersect(node) or prune_by_node(node):
                return

            if node.is_leaf:
                # 叶子: 搜索局部IVF
                if node.ivf_centroids is None:
                    # 暴力
                    for i, idx in enumerate(node.indices):
                        if not scalar_match(self.scalars[idx]):
                            continue
                        dist = np.linalg.norm(node.ivf_vectors[i] - query_vector)
                        update_heap(dist, idx)
                else:
                    # IVF: 探测最近的簇
                    centroid_dists = np.linalg.norm(
                        node.ivf_centroids - query_vector, axis=1
                    )
                    nearest = np.argsort(centroid_dists)[
                        : min(probe_clusters, len(node.ivf_centroids))
                    ]
                    for cid in nearest:
                        mask = node.ivf_labels == cid
                        for lid in np.where(mask)[0]:
                            idx = node.indices[lid]
                            if not scalar_match(self.scalars[idx]):
                                continue
                            dist = np.linalg.norm(node.ivf_vectors[lid] - query_vector)
                            update_heap(dist, idx)
            else:
                # 非叶子: 均值路由 (方案1)
                dist_l = np.linalg.norm(query_vector - node.left.mean_vector)
                dist_r = np.linalg.norm(query_vector - node.right.mean_vector)
                primary = node.left if dist_l <= dist_r else node.right
                secondary = node.right if dist_l <= dist_r else node.left

                recurse(primary)
                if not prune_by_node(secondary):
                    recurse(secondary)

        recurse(self.root)
        return sorted([idx for _, idx in heap])


def build_kdivf(vectors, scalars):
    """KDIVF 索引构建入口 (用于实验计时)"""
    return KDIVFIndex(vectors, scalars)


# ==========================================================
# 加载数据 (默认生成模拟数据，可切换为真实数据)
# ==========================================================

print("Loading dataset...")
# 若使用真实数据，修改 dataset_type 为 "npy" 或 "fvecs" 并配置 base_path
vector_data, query_vectors = load_datasets(
    dataset_type="generate",
    base_path="./random_ints_1m",
    n=max(DATA_SIZES),
    vector_dim=VECTOR_DIM,
)
print(f"Dataset shape: {vector_data.shape}")

# ==========================================================
# 实验主循环
# ==========================================================

results = []

for N in DATA_SIZES:
    print(f"\n===== Dataset Size {N} =====")

    vectors = vector_data[:N].copy()
    scalars = generate_scalars(N)

    # 1. KD-Tree (标量)
    t0 = time.perf_counter()
    build_kdtree(scalars)
    kd_time = time.perf_counter() - t0

    # 2. IVF (向量)
    t0 = time.perf_counter()
    build_ivf(vectors)
    ivf_time = time.perf_counter() - t0

    # 3. HNSW (向量)
    t0 = time.perf_counter()
    build_hnsw(vectors)
    hnsw_time = time.perf_counter() - t0

    # 4.  KDIVF (混合)
    t0 = time.perf_counter()
    build_kdivf(vectors, scalars)
    kdivf_time = time.perf_counter() - t0

    results.append([kd_time, ivf_time, hnsw_time, kdivf_time])

    print(f"  KD-Tree : {kd_time:.3f}s")
    print(f"  IVF     : {ivf_time:.3f}s")
    print(f"  HNSW    : {hnsw_time:.3f}s")
    print(f"  MRKD : {kdivf_time:.3f}s")

# ==========================================================
# 画图: 对比 [KD-Tree, IVF, HNSW, MRKD]
# ==========================================================

# 方法名称
methods = ["KD-Tree", "IVF", "HNSW", " MRKD"]

# 颜色 + 纹理
colors = ["#4C72B0", "#55A868", "#8172B3", "#C44E52"]  # 蓝, 绿, 紫, 红
hatches = ["//", "\\\\", "--", "xx"]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
bar_width = 0.18

for i, N in enumerate(DATA_SIZES):
    ax = axes[i]
    times = results[i]
    center = 0
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, 4)

    for j in range(len(methods)):
        ax.bar(
            center + offsets[j],
            times[j],
            width=bar_width,
            color=colors[j],
            hatch=hatches[j],
            edgecolor="black",
            label=methods[j] if i == 0 else "",
        )

    ax.set_title(f"{N//1000}K Data")
    ax.set_xticks([])
    ax.set_ylabel("Build Time (s)")
    ax.set_yscale("log")  # 对数轴更易看差异
    ax.grid(axis="y", linestyle="--", alpha=0.3)

axes[0].legend(frameon=False, loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("build_time_comparison.png", dpi=300, bbox_inches="tight")
print("\n✅ Plot saved to 'build_time_comparison.png'")
plt.show()
