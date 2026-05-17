import numpy as np
import time
import random
import json
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================================
# 参数配置
# ==========================================================

SCALAR_DIM = 5
VECTOR_DIM = 64

LEAF_SIZE = 200

IVF_CLUSTERS = 8
PROBE_CLUSTERS = 4

# β-WST 参数
BETA = 4

DATA_SIZES = [10000, 30000, 50000, 100000]

Q = 100
TOPK = 10

np.random.seed(42)
random.seed(42)

# ==========================================================
# 数据读取
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
# 数据加载
# ==========================================================


def load_datasets(
    dataset_type="generate",
    base_path="./random_ints_1m",
    n=100000,
    vector_dim=64,
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
# KDIVF
# KD-Tree + Leaf IVF
# ==========================================================


class KDIVFNode:

    def __init__(self, indices, scalars, vectors, depth=0):

        self.indices = indices

        self.is_leaf = len(indices) <= LEAF_SIZE

        # scalar bbox
        self.bbox_min = np.min(scalars[indices], axis=0)

        self.bbox_max = np.max(scalars[indices], axis=0)

        # vector statistics
        vecs = vectors[indices]

        self.mean_vector = np.mean(vecs, axis=0)

        self.max_radius = np.max(np.linalg.norm(vecs - self.mean_vector, axis=1))

        if self.is_leaf:

            self.axis = None
            self.split = None

            self.left = None
            self.right = None

            self._build_leaf_ivf(vectors[indices])

        else:

            self.axis = depth % scalars.shape[1]

            sorted_idx = indices[np.argsort(scalars[indices][:, self.axis])]

            mid = len(sorted_idx) // 2

            self.split = scalars[sorted_idx[mid]][self.axis]

            self.left = KDIVFNode(
                sorted_idx[:mid],
                scalars,
                vectors,
                depth + 1,
            )

            self.right = KDIVFNode(
                sorted_idx[mid:],
                scalars,
                vectors,
                depth + 1,
            )

            self.ivf_centroids = None
            self.ivf_labels = None
            self.ivf_vectors = None

    def _build_leaf_ivf(self, leaf_vectors):

        n = len(leaf_vectors)

        if n < IVF_CLUSTERS:

            self.ivf_centroids = None
            self.ivf_labels = None

        else:

            kmeans = KMeans(
                n_clusters=min(IVF_CLUSTERS, n),
                n_init=5,
                max_iter=50,
                random_state=42,
            )

            kmeans.fit(leaf_vectors)

            self.ivf_centroids = kmeans.cluster_centers_

            self.ivf_labels = kmeans.labels_

        self.ivf_vectors = leaf_vectors


class KDIVFIndex:

    def __init__(self, vectors, scalars):

        self.vectors = vectors

        self.scalars = scalars

        self.root = KDIVFNode(
            np.arange(len(vectors)),
            scalars,
            vectors,
        )


def build_kdivf(vectors, scalars):

    return KDIVFIndex(vectors, scalars)


# ==========================================================
# β-WST
# Multi-way Segment Tree + Leaf IVF
# ==========================================================


class BetaWSTNode:

    def __init__(self, indices, scalars, vectors, depth=0):

        self.indices = indices

        self.children = []

        self.is_leaf = len(indices) <= LEAF_SIZE

        # scalar bbox
        self.bbox_min = np.min(scalars[indices], axis=0)

        self.bbox_max = np.max(scalars[indices], axis=0)

        # vector statistics
        vecs = vectors[indices]

        self.mean_vector = np.mean(vecs, axis=0)

        self.max_radius = np.max(np.linalg.norm(vecs - self.mean_vector, axis=1))

        if self.is_leaf:

            self._build_leaf_ivf(vectors[indices])

            return

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
                scalars,
                vectors,
                depth + 1,
            )

            self.children.append(child)

        self.ivf_centroids = None
        self.ivf_labels = None
        self.ivf_vectors = None

    def _build_leaf_ivf(self, leaf_vectors):

        n = len(leaf_vectors)

        if n < IVF_CLUSTERS:

            self.ivf_centroids = None
            self.ivf_labels = None

        else:

            kmeans = KMeans(
                n_clusters=min(IVF_CLUSTERS, n),
                n_init=5,
                max_iter=50,
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
            scalars,
            vectors,
        )


def build_beta_wst(vectors, scalars):

    return BetaWSTIndex(vectors, scalars)


# ==========================================================
# 加载数据
# ==========================================================

print("Loading dataset...")

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

    row = []

    # ======================================================
    # KDIVF
    # ======================================================

    t0 = time.perf_counter()

    build_kdivf(vectors, scalars)

    kdivf_time = time.perf_counter() - t0

    row.append(kdivf_time)

    print(f"  KDIVF   : {kdivf_time:.3f}s")

    # ======================================================
    # β-WST
    # ======================================================

    t0 = time.perf_counter()

    build_beta_wst(vectors, scalars)

    beta_time = time.perf_counter() - t0

    row.append(beta_time)

    print(f"  β-WST   : {beta_time:.3f}s")

    results.append(row)

# ==========================================================
# 绘图
# ==========================================================

methods = ["KDIVF", "β-WST"]

colors = ["#4C72B0", "#C44E52"]

hatches = ["//", "xx"]

fig, axes = plt.subplots(1, len(DATA_SIZES), figsize=(18, 4))

bar_width = 0.25

for i, N in enumerate(DATA_SIZES):

    ax = axes[i]

    times = results[i]

    center = 0

    offsets = np.linspace(
        -0.5 * bar_width,
        0.5 * bar_width,
        len(methods),
    )

    for j in range(len(methods)):

        ax.bar(
            center + offsets[j],
            times[j],
            width=bar_width,
            color=colors[j],
            hatch=hatches[j],
            edgecolor="black",
            linewidth=0.8,
            label=methods[j] if i == 0 else "",
        )

        ax.text(
            center + offsets[j],
            times[j] * 1.05,
            f"{times[j]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(f"{N//1000}K Data")

    ax.set_xticks([])

    ax.set_ylabel("Build Time (s)")

    ax.set_yscale("log")

    ax.grid(axis="y", linestyle="--", alpha=0.3)

axes[0].legend(
    frameon=False,
    loc="upper left",
    fontsize=10,
)

plt.tight_layout()

plt.savefig(
    "kdivf_vs_beta_wst_build_time.png",
    dpi=300,
    bbox_inches="tight",
)

print("\n✅ Plot saved to 'kdivf_vs_beta_wst_build_time.png'")

plt.show()
