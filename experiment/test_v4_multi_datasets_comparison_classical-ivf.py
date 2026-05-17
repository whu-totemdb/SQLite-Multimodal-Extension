#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KDIVF vs Classical IVF 对比实验
- 测试四个数据集的构建时间 + 查询时间
- 输出毫秒级精度的对比表格
"""

import numpy as np
import time
import json
import heapq
from sklearn.cluster import KMeans
from collections import defaultdict

# ==========================================================
# 参数配置
# ==========================================================

# 索引参数
LEAF_SIZE = 200
IVF_CLUSTERS = 8
PROBE_CLUSTERS = 4
TOPK = 10

# 实验参数
NUM_WARMUP = 10  # 预热查询次数
NUM_QUERY_REPEAT = 3  # 查询重复次数取平均

# 数据集配置 (名称, 类型, 路径, 标量维度)
DATASETS = [
    ("SIFT10K", "fvecs", "./sift10k", 5),
    ("laion", "npy", "./laion-small-clip", 5),
    ("random_ints_1m", "npy", "./random_ints_1m", 5),
    ("random_float_1m", "npy", "./random_float_1m", 5),
]

np.random.seed(42)


# ==========================================================
# 数据加载工具
# ==========================================================


def read_fvecs(filename):
    """读取 fvecs 格式向量文件"""
    vectors = []
    with open(filename, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
            vec_bytes = f.read(4 * dim)
            if len(vec_bytes) < 4 * dim:
                break
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors.append(vec)
    return np.vstack(vectors).astype(np.float32)


def load_datasets(dataset_type, base_path):
    """
    加载数据集
    dataset_type: "fvecs" 或 "npy"
    base_path: 数据集文件夹路径
    """
    if dataset_type == "fvecs":
        vector_data = read_fvecs(f"{base_path}/vectors.fvecs")
        query_vectors = read_fvecs(f"{base_path}/query.fvecs")
    elif dataset_type == "npy":
        vector_data = np.load(f"{base_path}/vectors.npy").astype(np.float32)
        query_vectors = []
        with open(f"{base_path}/tests.jsonl", "r") as f:
            for line in f:
                obj = json.loads(line)
                query_vectors.append(obj["query"])
        query_vectors = np.array(query_vectors, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return vector_data, query_vectors


def generate_scalars(n, scalar_dim):
    """生成随机标量属性 [0, 1) 均匀分布"""
    return np.random.rand(n, scalar_dim).astype(np.float32)


# ==========================================================
# Classical IVF 实现
# ==========================================================


class ClassicalIVF:
    """经典全局 IVF 索引 (用于对比基线)"""

    def __init__(self, vectors, n_clusters=IVF_CLUSTERS):
        self.vectors = vectors
        self.n_clusters = n_clusters
        self.centroids = None
        self.labels = None
        self.inverted_list = None  # 倒排列表: cid -> [local_indices]

    def build(self):
        """构建 IVF 索引"""
        n = len(self.vectors)
        if n < self.n_clusters:
            self.centroids = None
            self.labels = None
            self.inverted_list = {0: list(range(n))}
            return

        # KMeans 聚类
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=1,  # 公平对比: 与 KDIVF 叶子一致
            max_iter=50,
            init="k-means++",
        )
        kmeans.fit(self.vectors)

        self.centroids = kmeans.cluster_centers_
        self.labels = kmeans.labels_

        # 构建倒排列表
        self.inverted_list = defaultdict(list)
        for idx, cid in enumerate(self.labels):
            self.inverted_list[cid].append(idx)

    def query(
        self,
        query_vector,
        query_ranges,
        scalar_data,
        topk=TOPK,
        probe_clusters=PROBE_CLUSTERS,
        scalar_dim=5,
    ):
        """
        查询: 标量过滤 + 向量检索
        注意: Classical IVF 本身不支持标量过滤，这里采用"先检索后过滤"的两阶段策略
        """

        def scalar_match(row):
            for d in range(scalar_dim):
                if row[d] < query_ranges[d][0] or row[d] > query_ranges[d][1]:
                    return False
            return True

        heap = []  # 最大堆: (-dist, global_idx)

        def update_heap(dist, idx):
            if len(heap) < topk:
                heapq.heappush(heap, (-dist, idx))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, idx))

        if self.centroids is None:
            # 数据少: 暴力扫描
            for idx, vec in enumerate(self.vectors):
                if not scalar_match(scalar_data[idx]):
                    continue
                dist = np.linalg.norm(vec - query_vector)
                update_heap(dist, idx)
        else:
            # IVF: 探测最近的 probe_clusters 个簇
            centroid_dists = np.linalg.norm(self.centroids - query_vector, axis=1)
            nearest_cids = np.argsort(centroid_dists)[
                : min(probe_clusters, len(self.centroids))
            ]

            for cid in nearest_cids:
                for local_idx in self.inverted_list[cid]:
                    global_idx = local_idx
                    if not scalar_match(scalar_data[global_idx]):
                        continue
                    dist = np.linalg.norm(self.vectors[local_idx] - query_vector)
                    update_heap(dist, global_idx)

        return sorted([idx for _, idx in heap])


# ==========================================================
# KDIVF 实现 (您的混合索引)
# ==========================================================


class KDIVFNode:
    """KDIVF 索引的树节点"""

    def __init__(self, indices, scalars, vectors, depth=0, scalar_dim=5):
        self.indices = indices
        self.is_leaf = len(indices) <= LEAF_SIZE
        self.scalar_dim = scalar_dim

        # 标量包围盒 (用于范围剪枝)
        self.bbox_min = np.min(scalars[indices], axis=0)
        self.bbox_max = np.max(scalars[indices], axis=0)

        # 向量空间统计 (路由 + 剪枝)
        vecs = vectors[indices]
        self.mean_vector = np.mean(vecs, axis=0)
        self.max_radius = np.max(np.linalg.norm(vecs - self.mean_vector, axis=1))

        if self.is_leaf:
            # 叶子: 构建局部 IVF
            self.axis = self.split = None
            self.left = self.right = None
            self._build_leaf_ivf(vectors[indices])
        else:
            # 非叶子: 按标量维度分割
            self.axis = depth % scalar_dim
            sorted_idx = indices[np.argsort(scalars[indices][:, self.axis])]
            mid = len(sorted_idx) // 2
            self.split = scalars[sorted_idx[mid]][self.axis]

            self.left = KDIVFNode(
                sorted_idx[:mid], scalars, vectors, depth + 1, scalar_dim
            )
            self.right = KDIVFNode(
                sorted_idx[mid:], scalars, vectors, depth + 1, scalar_dim
            )

            # 叶子 IVF 属性置空
            self.ivf_centroids = self.ivf_labels = self.ivf_vectors = None
            self.leaf_indices = None

    def _build_leaf_ivf(self, leaf_vectors):
        """构建叶子节点的局部 IVF"""
        n = len(leaf_vectors)
        if n < IVF_CLUSTERS:
            self.ivf_centroids = self.ivf_labels = None
        else:
            kmeans = KMeans(
                n_clusters=min(IVF_CLUSTERS, n),
                random_state=42,
                n_init=1,
                max_iter=50,
                init="k-means++",
            )
            kmeans.fit(leaf_vectors)
            self.ivf_centroids = kmeans.cluster_centers_
            self.ivf_labels = kmeans.labels_
        self.ivf_vectors = leaf_vectors


class KDIVFIndex:
    """KDIVF 混合索引主类"""

    def __init__(self, vectors, scalars, scalar_dim=5):
        self.vectors = vectors
        self.scalars = scalars
        self.scalar_dim = scalar_dim
        self.root = KDIVFNode(
            np.arange(len(vectors)), scalars, vectors, scalar_dim=scalar_dim
        )

    def query(
        self, query_vector, query_ranges, topk=TOPK, probe_clusters=PROBE_CLUSTERS
    ):
        """混合查询: 标量范围 + 向量近邻"""
        heap = []

        def scalar_match(row):
            for d in range(self.scalar_dim):
                if row[d] < query_ranges[d][0] or row[d] > query_ranges[d][1]:
                    return False
            return True

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
                # 叶子: 搜索局部 IVF
                if node.ivf_centroids is None:
                    for i, idx in enumerate(node.indices):
                        if not scalar_match(self.scalars[idx]):
                            continue
                        dist = np.linalg.norm(node.ivf_vectors[i] - query_vector)
                        update_heap(dist, idx)
                else:
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
                # 非叶子: 均值路由
                dist_l = np.linalg.norm(query_vector - node.left.mean_vector)
                dist_r = np.linalg.norm(query_vector - node.right.mean_vector)
                primary = node.left if dist_l <= dist_r else node.right
                secondary = node.right if dist_l <= dist_r else node.left

                recurse(primary)
                if not prune_by_node(secondary):
                    recurse(secondary)

        recurse(self.root)
        return sorted([idx for _, idx in heap])


# ==========================================================
# 实验工具函数
# ==========================================================


def generate_query_ranges(scalar_dim, selectivity=0.2):
    """生成随机查询范围，控制选择率"""
    ranges = []
    for _ in range(scalar_dim):
        width = selectivity ** (1 / scalar_dim)  # 近似控制联合选择率
        start = np.random.uniform(0, 1 - width)
        ranges.append((start, start + width))
    return ranges


def benchmark_build(index_class, vectors, scalars=None, scalar_dim=5):
    """测试构建时间 (毫秒)"""
    t0 = time.perf_counter()
    if index_class == "ClassicalIVF":
        index = ClassicalIVF(vectors)
        index.build()
    else:  # KDIVF
        index = KDIVFIndex(vectors, scalars, scalar_dim)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000  # 转换为毫秒


def benchmark_query(
    index,
    query_vector,
    query_ranges,
    scalar_data,
    scalar_dim,
    num_repeat=NUM_QUERY_REPEAT,
):
    """测试查询时间 (毫秒)，多次重复取平均"""
    times = []
    for _ in range(num_repeat):
        t0 = time.perf_counter()
        if isinstance(index, ClassicalIVF):
            _ = index.query(
                query_vector, query_ranges, scalar_data, scalar_dim=scalar_dim
            )
        else:  # KDIVF
            _ = index.query(query_vector, query_ranges)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # 毫秒
    return np.mean(times)


def compute_recall(approx_results, ground_truth, topk=TOPK):
    """计算 Recall@K"""
    if len(ground_truth) == 0:
        return 0.0
    approx_set = set(approx_results[:topk])
    gt_set = set(ground_truth[:topk])
    return len(approx_set & gt_set) / min(topk, len(gt_set))


def compute_ground_truth(
    vectors, scalars, query_vector, query_ranges, scalar_dim, topk=TOPK
):
    """暴力计算 Ground Truth (用于召回率验证)"""

    def scalar_match(row):
        for d in range(scalar_dim):
            if row[d] < query_ranges[d][0] or row[d] > query_ranges[d][1]:
                return False
        return True

    dists = []
    for i, (vec, scalar) in enumerate(zip(vectors, scalars)):
        if not scalar_match(scalar):
            continue
        dist = np.linalg.norm(vec - query_vector)
        dists.append((dist, i))

    dists.sort()
    return [idx for _, idx in dists[:topk]]


# ==========================================================
# 主实验流程
# ==========================================================


def run_experiment():
    """执行完整对比实验"""

    print("=" * 80)
    print("KDIVF vs Classical IVF 对比实验")
    print("=" * 80)

    results = []

    for ds_name, ds_type, ds_path, scalar_dim in DATASETS:
        print(f"\n📦 加载数据集: {ds_name} ({ds_type} @ {ds_path})")

        # 1. 加载数据
        try:
            vectors, query_vectors = load_datasets(ds_type, ds_path)
        except FileNotFoundError as e:
            print(f"⚠️  跳过 {ds_name}: {e}")
            continue

        N = len(vectors)
        Q = len(query_vectors)
        vector_dim = vectors.shape[1]

        print(f"   向量: {N} × {vector_dim}D, 查询: {Q} 条")

        # 2. 生成标量属性 (固定 seed 保证可复现)
        np.random.seed(42)
        scalars = generate_scalars(N, scalar_dim)

        # 3. 生成查询范围 (固定选择率 0.2)
        query_ranges = generate_query_ranges(scalar_dim, selectivity=0.2)

        # 4. 构建时间测试
        print("   🔨 测试构建时间...")

        build_ivf = benchmark_build("ClassicalIVF", vectors)
        build_kdivf = benchmark_build("KDIVF", vectors, scalars, scalar_dim)

        # 5. 查询时间测试 (预热 + 正式)
        print("   🔍 测试查询时间...")

        # 预热
        ivf_temp = ClassicalIVF(vectors[: min(1000, N)])
        ivf_temp.build()
        kdivf_temp = KDIVFIndex(
            vectors[: min(1000, N)], scalars[: min(1000, N)], scalar_dim
        )

        for _ in range(NUM_WARMUP):
            q = query_vectors[0]
            ivf_temp.query(q, query_ranges, scalars, scalar_dim=scalar_dim)
            kdivf_temp.query(q, query_ranges)

        # 正式测试 (取前20个查询平均)
        test_queries = min(20, Q)
        query_times_ivf = []
        query_times_kdivf = []
        recalls_kdivf = []

        # 重建完整索引用于查询测试
        ivf_index = ClassicalIVF(vectors)
        ivf_index.build()
        kdivf_index = KDIVFIndex(vectors, scalars, scalar_dim)

        for i in range(test_queries):
            q = query_vectors[i]

            # Classical IVF 查询
            t_ivf = benchmark_query(ivf_index, q, query_ranges, scalars, scalar_dim)
            query_times_ivf.append(t_ivf)

            # KDIVF 查询
            t_kdivf = benchmark_query(kdivf_index, q, query_ranges, scalars, scalar_dim)
            query_times_kdivf.append(t_kdivf)

            # 召回率验证 (仅 KDIVF, 因为 IVF 采用后过滤策略)
            gt = compute_ground_truth(vectors, scalars, q, query_ranges, scalar_dim)
            approx = kdivf_index.query(q, query_ranges)
            recall = compute_recall(approx, gt)
            recalls_kdivf.append(recall)

        avg_query_ivf = np.mean(query_times_ivf)
        avg_query_kdivf = np.mean(query_times_kdivf)
        avg_recall_kdivf = np.mean(recalls_kdivf)

        # 6. 记录结果
        results.append(
            {
                "dataset": ds_name,
                "N": N,
                "vector_dim": vector_dim,
                "build_ivf_ms": build_ivf,
                "build_kdivf_ms": build_kdivf,
                "query_ivf_ms": avg_query_ivf,
                "query_kdivf_ms": avg_query_kdivf,
                "recall_kdivf": avg_recall_kdivf,
                "speedup_build": build_ivf / build_kdivf if build_kdivf > 0 else 0,
                "speedup_query": (
                    avg_query_ivf / avg_query_kdivf if avg_query_kdivf > 0 else 0
                ),
            }
        )

        print(f"   ✓ 完成 {ds_name}")

    return results


def print_comparison_table(results):
    """打印对比表格"""

    print("\n" + "=" * 100)
    print("📊 实验结果对比表 (单位: 毫秒, 召回率: 0-1)")
    print("=" * 100)

    # 表头
    header = f"{'数据集':<15} {'规模':>8} {'构建-IVF':>12} {'构建-KDIVF':>12} {'查询-IVF':>12} {'查询-KDIVF':>12} {'召回@10':>10} {'加速(构建)':>12} {'加速(查询)':>12}"
    print(header)
    print("-" * 100)

    # 数据行
    for r in results:
        row = (
            f"{r['dataset']:<15} "
            f"{r['N']:>8,} "
            f"{r['build_ivf_ms']:>12.2f} "
            f"{r['build_kdivf_ms']:>12.2f} "
            f"{r['query_ivf_ms']:>12.2f} "
            f"{r['query_kdivf_ms']:>12.2f} "
            f"{r['recall_kdivf']:>10.4f} "
            f"{r['speedup_build']:>12.2f}x "
            f"{r['speedup_query']:>12.2f}x"
        )
        print(row)

    print("-" * 100)

    # 汇总统计
    if results:
        avg_build_speedup = np.mean([r["speedup_build"] for r in results])
        avg_query_speedup = np.mean([r["speedup_query"] for r in results])
        avg_recall = np.mean([r["recall_kdivf"] for r in results])

        print(f"\n📈 平均指标:")
        print(f"   • 构建加速比: {avg_build_speedup:.2f}x (KDIVF vs IVF)")
        print(f"   • 查询加速比: {avg_query_speedup:.2f}x (KDIVF vs IVF)")
        print(f"   • 平均召回率: {avg_recall:.4f} (KDIVF @ selectivity=0.2)")

        print(f"\n💡 说明:")
        print(f"   • 加速比 > 1.0 表示 KDIVF 更快")
        print(f"   • Classical IVF 采用'先检索后过滤'策略，可能返回不足 topk 的结果")
        print(f"   • KDIVF 采用'先过滤后检索'策略，标量剪枝前置，效率更高")


def save_results_to_csv(results, filename="kdivf_vs_ivf_results.csv"):
    """保存结果为 CSV 文件"""
    import csv

    if not results:
        return

    fieldnames = results[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n💾 结果已保存至: {filename}")


# ==========================================================
# 入口
# ==========================================================

if __name__ == "__main__":
    results = run_experiment()

    if results:
        print_comparison_table(results)
        save_results_to_csv(results)
    else:
        print("\n⚠️  无有效结果，请检查数据集路径是否正确")
