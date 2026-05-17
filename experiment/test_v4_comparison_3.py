import numpy as np
import heapq
import time
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ==============================
# 核心配置
# ==============================
DATA_SIZES = [10000, 30000, 50000]
DATA_LABELS = ["1W", "3W", "5W"]
VECTOR_DIM = 64
SCALAR_DIM = 5
QUERY_NUM = 50
TOPK = 10

LEAF_SIZE = 200
IVF_CLUSTERS = 8
PROBE_CLUSTERS = 16
HNSW_M = 8
HNSW_CANDIDATES = 200
HNSW_MAX_VISIT = 2000

np.random.seed(42)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]  # 中文+英文兼容
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方框的问题
plt.rcParams["font.size"] = 13


# ==============================
# 工具函数
# ==============================
def scalar_filter(index, scalar_data, query_ranges):
    val = scalar_data[index]
    return np.all((val >= query_ranges[:, 0]) & (val <= query_ranges[:, 1]))


def update_topk_heap(heap, dist, idx):
    if len(heap) < TOPK:
        heapq.heappush(heap, (-dist, idx))
    elif dist < -heap[0][0]:
        heapq.heappushpop(heap, (-dist, idx))


def calculate_recall(predict_indices, ground_truth_set):
    predict_set = set(predict_indices)
    return len(predict_set & ground_truth_set) / TOPK


def generate_dataset(N):
    vector_data = np.random.randn(N, VECTOR_DIM).astype(np.float32)
    scalar_data = np.random.rand(N, SCALAR_DIM).astype(np.float32)
    query_vectors = np.random.randn(QUERY_NUM, VECTOR_DIM).astype(np.float32)
    query_ranges = np.array([(0.2, 0.8)] * SCALAR_DIM)
    return vector_data, scalar_data, query_vectors, query_ranges


def get_ground_truth(vector_data, scalar_data, query_vectors, query_ranges):
    gt_list = []
    N = len(vector_data)
    for q in query_vectors:
        heap = []
        for i in range(N):
            if not scalar_filter(i, scalar_data, query_ranges):
                continue
            dist = np.linalg.norm(vector_data[i] - q)
            update_topk_heap(heap, dist, i)
        gt_indices = [idx for (_, idx) in heap]
        gt_list.append(set(gt_indices))
    return gt_list


def get_size_mb(obj):
    if isinstance(obj, np.ndarray):
        return obj.nbytes / (1024 * 1024)
    if isinstance(obj, dict):
        total = 0
        for v in obj.values():
            total += get_size_mb(v)
        return total
    if isinstance(obj, list):
        total = 0
        for item in obj:
            total += get_size_mb(item)
        return total
    if hasattr(obj, "__dict__"):
        return get_size_mb(obj.__dict__)
    return sys.getsizeof(obj) / (1024 * 1024)


# ==============================
# 索引实现
# ==============================
class IVFScalarIndex:
    def __init__(self, vector_data):
        self.vector_data = vector_data
        start = time.time()
        self.kmeans = KMeans(n_clusters=IVF_CLUSTERS, random_state=42, n_init=5)
        self.kmeans.fit(vector_data)
        self.build_time = time.time() - start
        self.memory_mb = get_size_mb(self.kmeans.cluster_centers_) + get_size_mb(
            self.kmeans.labels_
        )

    def query(self, q_vec, scalar_data, q_ranges):
        heap = []
        dists = np.linalg.norm(self.kmeans.cluster_centers_ - q_vec, axis=1)
        top_cids = np.argsort(dists)[:PROBE_CLUSTERS]
        for cid in top_cids:
            ids = np.where(self.kmeans.labels_ == cid)[0]
            for i in ids:
                if scalar_filter(i, scalar_data, q_ranges):
                    d = np.linalg.norm(self.vector_data[i] - q_vec)
                    update_topk_heap(heap, d, i)
        return [i for (_, i) in heap]


class HNSWScalarIndex:
    def __init__(self, vector_data):
        self.vector_data = vector_data
        N = len(vector_data)
        start = time.time()
        self.neighbors = {i: [] for i in range(N)}
        for i in range(N):
            sample_ids = np.random.choice(N, HNSW_CANDIDATES, replace=False)
            dists = np.linalg.norm(vector_data[sample_ids] - vector_data[i], axis=1)
            nearest = sample_ids[np.argsort(dists)[1 : HNSW_M + 1]]
            self.neighbors[i] = nearest
        self.build_time = time.time() - start
        self.memory_mb = get_size_mb(self.neighbors)

    def query(self, q_vec, scalar_data, q_ranges):
        heap = []
        visited = set()
        stack = [np.random.randint(len(self.vector_data))]
        max_visit = HNSW_MAX_VISIT
        while stack and len(visited) < max_visit:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            if scalar_filter(u, scalar_data, q_ranges):
                d = np.linalg.norm(self.vector_data[u] - q_vec)
                update_topk_heap(heap, d, u)
            for v in self.neighbors[u]:
                if v not in visited:
                    stack.append(v)
        return [i for (_, i) in heap]


class KDLeafIVFIndex:
    def __init__(self, vector_data, scalar_data):
        self.vector_data = vector_data
        self.scalar_data = scalar_data
        start = time.time()

        class KDNode:
            __slots__ = [
                "axis",
                "left",
                "right",
                "bbox_min",
                "bbox_max",
                "is_leaf",
                "indices",
                "mean_vec",
                "max_radius",
                "leaf_id",
            ]

            def __init__(self, indices, depth=0):
                self.left = self.right = None
                self.axis = depth % SCALAR_DIM
                self.is_leaf = len(indices) <= LEAF_SIZE
                self.bbox_min = np.min(scalar_data[indices], axis=0)
                self.bbox_max = np.max(scalar_data[indices], axis=0)
                vecs = vector_data[indices]
                self.mean_vec = np.mean(vecs, axis=0)
                self.max_radius = np.max(np.linalg.norm(vecs - self.mean_vec, axis=1))
                self.indices = indices if self.is_leaf else None
                self.leaf_id = -1
                if not self.is_leaf:
                    sorted_idx = indices[np.argsort(scalar_data[indices][:, self.axis])]
                    mid = len(sorted_idx) // 2
                    self.left = KDNode(sorted_idx[:mid], depth + 1)
                    self.right = KDNode(sorted_idx[mid:], depth + 1)

        self.root = KDNode(np.arange(len(vector_data)))
        self.leaves = []

        def collect(node):
            if node.is_leaf:
                node.leaf_id = len(self.leaves)
                self.leaves.append(node)
            else:
                collect(node.left)
                collect(node.right)

        collect(self.root)

        self.ivf = {}
        for node in self.leaves:
            vecs = vector_data[node.indices]
            if len(vecs) < IVF_CLUSTERS:
                self.ivf[node.leaf_id] = {"cent": None, "vecs": vecs}
                continue
            km = KMeans(n_clusters=IVF_CLUSTERS, random_state=42, n_init=5)
            km.fit(vecs)
            self.ivf[node.leaf_id] = {
                "cent": km.cluster_centers_,
                "labels": km.labels_,
                "vecs": vecs,
            }
        self.build_time = time.time() - start
        self.memory_mb = get_size_mb(self.root) + get_size_mb(self.ivf)

    def query(self, q_vec):
        heap = []
        q_ranges = np.array([(0.2, 0.8)] * SCALAR_DIM)

        def bbox_intersect(node):
            return np.all(
                (node.bbox_max >= q_ranges[:, 0]) & (node.bbox_min <= q_ranges[:, 1])
            )

        def prune(node):
            if len(heap) < TOPK:
                return False
            dist = np.linalg.norm(q_vec - node.mean_vec)
            return dist - node.max_radius > -heap[0][0]

        def recurse(node):
            if not bbox_intersect(node) or prune(node):
                return
            if node.is_leaf:
                info = self.ivf[node.leaf_id]
                cents, vecs = info["cent"], info["vecs"]
                if cents is None:
                    for i, idx in enumerate(node.indices):
                        if scalar_filter(idx, self.scalar_data, q_ranges):
                            d = np.linalg.norm(vecs[i] - q_vec)
                            update_topk_heap(heap, d, idx)
                    return
                dists = np.linalg.norm(cents - q_vec, axis=1)
                for c in np.argsort(dists)[:PROBE_CLUSTERS]:
                    for lid in np.where(info["labels"] == c)[0]:
                        idx = node.indices[lid]
                        if scalar_filter(idx, self.scalar_data, q_ranges):
                            d = np.linalg.norm(vecs[lid] - q_vec)
                            update_topk_heap(heap, d, idx)
                return
            if np.linalg.norm(q_vec - node.left.mean_vec) < np.linalg.norm(
                q_vec - node.right.mean_vec
            ):
                recurse(node.left)
                if not prune(node.right):
                    recurse(node.right)
            else:
                recurse(node.right)
                if not prune(node.left):
                    recurse(node.left)

        recurse(self.root)
        return [i for (_, i) in heap]


# ==============================
# 评估函数
# ==============================
def evaluate_index(index, query_vectors, ground_truth, scalar_data=None, q_ranges=None):
    total_recall = 0.0
    total_query_time = 0.0
    for q, gt in zip(query_vectors, ground_truth):
        s = time.time()
        if isinstance(index, KDLeafIVFIndex):
            pred = index.query(q)
        else:
            pred = index.query(q, scalar_data, q_ranges)
        total_query_time += time.time() - s
        total_recall += calculate_recall(pred, gt)
    avg_recall = total_recall / QUERY_NUM
    avg_q_time = total_query_time / QUERY_NUM
    return avg_recall, avg_q_time


# ==============================
# 绘图函数（已优化：组内紧贴，组间留半柱空隙）
# ==============================
def plot_build_time(results):
    """独立图1：构建时间"""
    # ========== 样式配置（你在这里随便改） ==========
    colors = ["#4472C4", "#ED7D31", "#70AD47"]  # 颜色
    hatches = ["///", "\\\\\\", "xxx"]  # 纹理
    index_names = ["IVF-Scalar", "HNSW-Scalar", "MRKD"]
    bar_width = 0.25  # 柱子宽度
    group_gap = bar_width * 0.5  # 组间空隙 = 半个柱子

    x_base = np.arange(len(DATA_LABELS)) * (bar_width * 3 + group_gap)  # 核心：组间距

    plt.figure(figsize=(10, 6))
    ivf_build = [r["ivf_build"] for r in results]
    hnsw_build = [r["hnsw_build"] for r in results]
    yours_build = [r["yours_build"] for r in results]

    plt.bar(
        x_base - bar_width,
        ivf_build,
        bar_width,
        label=index_names[0],
        color=colors[0],
        hatch=hatches[0],
    )
    plt.bar(
        x_base,
        hnsw_build,
        bar_width,
        label=index_names[1],
        color=colors[1],
        hatch=hatches[1],
    )
    plt.bar(
        x_base + bar_width,
        yours_build,
        bar_width,
        label=index_names[2],
        color=colors[2],
        hatch=hatches[2],
    )

    plt.title("索引构建时间对比图", fontsize=16, fontweight="bold")
    plt.xlabel("Data Size", fontsize=14)
    plt.ylabel("Build Time (s)", fontsize=14)
    plt.xticks(x_base, DATA_LABELS)
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=12)
    for bars in plt.gca().containers:
        plt.bar_label(bars, fmt="%.2f", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("index_build_time.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_memory_usage(results):
    """独立图2：内存占用"""
    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    hatches = ["///", "\\\\\\", "xxx"]
    index_names = ["IVF-Scalar", "HNSW-Scalar", "MRKD"]
    bar_width = 0.25
    group_gap = bar_width * 0.5
    x_base = np.arange(len(DATA_LABELS)) * (bar_width * 3 + group_gap)

    plt.figure(figsize=(10, 6))
    ivf_mem = [r["ivf_mem"] for r in results]
    hnsw_mem = [r["hnsw_mem"] for r in results]
    yours_mem = [r["yours_mem"] for r in results]

    plt.bar(
        x_base - bar_width,
        ivf_mem,
        bar_width,
        label=index_names[0],
        color=colors[0],
        hatch=hatches[0],
    )
    plt.bar(
        x_base,
        hnsw_mem,
        bar_width,
        label=index_names[1],
        color=colors[1],
        hatch=hatches[1],
    )
    plt.bar(
        x_base + bar_width,
        yours_mem,
        bar_width,
        label=index_names[2],
        color=colors[2],
        hatch=hatches[2],
    )

    plt.title("索引内存占用对比图", fontsize=16, fontweight="bold")
    plt.xlabel("Data Size", fontsize=14)
    plt.ylabel("Memory (MB)", fontsize=14)
    plt.xticks(x_base, DATA_LABELS)
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=12)
    for bars in plt.gca().containers:
        plt.bar_label(bars, fmt="%.2f", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("index_memory_usage.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_query_latency(results):
    """独立图3：查询耗时"""
    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    hatches = ["///", "\\\\\\", "xxx"]
    index_names = ["IVF-Scalar", "HNSW-Scalar", "MRKD"]
    bar_width = 0.25
    group_gap = bar_width * 0.5
    x_base = np.arange(len(DATA_LABELS)) * (bar_width * 3 + group_gap)

    plt.figure(figsize=(10, 6))
    ivf_query = [r["ivf_query"] for r in results]
    hnsw_query = [r["hnsw_query"] for r in results]
    yours_query = [r["yours_query"] for r in results]

    plt.bar(
        x_base - bar_width,
        ivf_query,
        bar_width,
        label=index_names[0],
        color=colors[0],
        hatch=hatches[0],
    )
    plt.bar(
        x_base,
        hnsw_query,
        bar_width,
        label=index_names[1],
        color=colors[1],
        hatch=hatches[1],
    )
    plt.bar(
        x_base + bar_width,
        yours_query,
        bar_width,
        label=index_names[2],
        color=colors[2],
        hatch=hatches[2],
    )

    plt.title("索引查询延迟对比图", fontsize=16, fontweight="bold")
    plt.xlabel("Data Size", fontsize=14)
    plt.ylabel("Query Latency (ms)", fontsize=14)
    plt.xticks(x_base, DATA_LABELS)
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=12)
    for bars in plt.gca().containers:
        plt.bar_label(bars, fmt="%.2f", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("index_query_latency.png", dpi=300, bbox_inches="tight")
    plt.show()


# ==============================
# 主流程
# ==============================
if __name__ == "__main__":
    results = []
    for idx, N in enumerate(DATA_SIZES):
        print(f"\n{'='*75}")
        print(f"                数据集规模：{N:,}")
        print(f"{'='*75}")
        vector_data, scal_data, q_vecs, q_ranges = generate_dataset(N)
        print("计算 Ground Truth...")
        gt = get_ground_truth(vector_data, scal_data, q_vecs, q_ranges)
        print("构建索引...")

        index1 = IVFScalarIndex(vector_data)
        index2 = HNSWScalarIndex(vector_data)
        indexY = KDLeafIVFIndex(vector_data, scal_data)

        print("评估查询性能...")
        r1, t1 = evaluate_index(index1, q_vecs, gt, scal_data, q_ranges)
        r2, t2 = evaluate_index(index2, q_vecs, gt, scal_data, q_ranges)
        ry, ty = evaluate_index(indexY, q_vecs, gt)

        res = {
            "ivf_build": index1.build_time,
            "ivf_mem": index1.memory_mb,
            "ivf_query": t1 * 1000,
            "hnsw_build": index2.build_time,
            "hnsw_mem": index2.memory_mb,
            "hnsw_query": t2 * 1000,
            "yours_build": indexY.build_time,
            "yours_mem": indexY.memory_mb,
            "yours_query": ty * 1000,
        }
        results.append(res)

        print("-" * 80)
        print(
            f"{'索引名称':<20} {'构建时间(s)':<10} {'内存(MB)':<10} {'查询(ms)':<10} {'Recall':<10}"
        )
        print("-" * 80)
        print(
            f"IVF-Scalar    | {index1.build_time:<8.2f} {index1.memory_mb:<8.2f} {t1*1000:<8.2f} {r1:.4f}"
        )
        print(
            f"HNSW-Scalar   | {index2.build_time:<8.2f} {index2.memory_mb:<8.2f} {t2*1000:<8.2f} {r2:.4f}"
        )
        print(
            f"MRKD        | {indexY.build_time:<8.2f} {indexY.memory_mb:<8.2f} {ty*1000:<8.2f} {ry:.4f}"
        )
        print("-" * 80)

    # 依次生成三张独立图
    print("\n📊 生成 构建时间 图...")
    plot_build_time(results)
    print("\n📊 生成 内存占用 图...")
    plot_memory_usage(results)
    print("\n📊 生成 查询耗时 图...")
    plot_query_latency(results)

    print("\n✅ 三张高清图片已全部保存！")
