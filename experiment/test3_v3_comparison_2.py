import numpy as np
import time
import sqlite3
import os
import psutil
import pandas as pd
import gc
import heapq

try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

# ====================== CONFIG ===========================
DATASET_SIZE = 30000
N_QUERY = 500
TOPK = 10
VECTOR_DIM = 64
np.random.seed(42)

# 生成模拟数据
xb_vec = np.random.randn(DATASET_SIZE, VECTOR_DIM).astype(np.float32)
xq_vec = np.random.randn(N_QUERY, VECTOR_DIM).astype(np.float32)

N = len(xb_vec)
DIM = VECTOR_DIM

def memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def percentile_latency(times, p):
    return np.percentile(times, p)

# ====================== SQLite 原生 =======================
def test_sqlite():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE vectors(id INTEGER PRIMARY KEY, vec BLOB)")
    start_build = time.perf_counter()
    for i in range(N):
        cur.execute("INSERT INTO vectors VALUES (?,?)", (i, xb_vec[i].tobytes()))
    conn.commit()
    build_time = time.perf_counter() - start_build

    # 查询
    query_times = []
    for q in xq_vec:
        start = time.perf_counter()
        rows = cur.execute("SELECT id, vec FROM vectors").fetchall()
        dists = []
        for row_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            dists.append((np.linalg.norm(vec - q), row_id))
        dists.sort(key=lambda x: x[0])
        _ = [idx for _, idx in dists[:TOPK]]
        query_times.append(time.perf_counter() - start)
    conn.close()

    return {
        "BuildTime": build_time,
        "AvgLatency": np.mean(query_times)*1000,
        "P95Latency": percentile_latency(query_times, 95)*1000,
        "QPS": 1/np.mean(query_times),
        "MemoryMB": memory_mb()
    }

# ====================== Hybrid KDTree+IVF =======================
class KDNode:
    def __init__(self, indices, depth=0, leaf_size=600):
        self.indices = indices
        self.left = None
        self.right = None
        self.is_leaf = len(indices) <= leaf_size
        self.axis = depth % VECTOR_DIM
        if not self.is_leaf:
            sorted_idx = indices[np.argsort(xb_vec[indices][:, self.axis])]
            mid = len(sorted_idx)//2
            self.left = KDNode(sorted_idx[:mid], depth+1, leaf_size)
            self.right = KDNode(sorted_idx[mid:], depth+1, leaf_size)

def search_kdtree(node, q, topk):
    heap = []
    def recurse(n):
        if n.is_leaf:
            for idx in n.indices:
                dist = np.linalg.norm(xb_vec[idx]-q)
                if len(heap)<topk:
                    heapq.heappush(heap, (-dist, idx))
                elif dist < -heap[0][0]:
                    heapq.heapreplace(heap, (-dist, idx))
        else:
            recurse(n.left)
            recurse(n.right)
    recurse(node)
    return [idx for _, idx in sorted(heap, key=lambda x: -x[0])]

def test_hybrid():
    start_build = time.perf_counter()
    root = KDNode(np.arange(N))
    build_time = time.perf_counter() - start_build

    query_times=[]
    for q in xq_vec:
        start = time.perf_counter()
        _ = search_kdtree(root, q, TOPK)
        query_times.append(time.perf_counter()-start)
    return {
        "BuildTime": build_time,
        "AvgLatency": np.mean(query_times)*1000,
        "P95Latency": percentile_latency(query_times,95)*1000,
        "QPS": 1/np.mean(query_times),
        "MemoryMB": memory_mb()
    }

# ====================== SQLite-VSS Flat =======================
def test_vss_flat():
    if not FAISS_AVAILABLE:
        return None
    start_build = time.perf_counter()
    index = faiss.IndexFlatL2(DIM)
    index.add(xb_vec)
    build_time = time.perf_counter()-start_build

    query_times=[]
    for q in xq_vec:
        start = time.perf_counter()
        D,I = index.search(q.reshape(1,-1), TOPK)
        query_times.append(time.perf_counter()-start)
    return {
        "BuildTime": build_time,
        "AvgLatency": np.mean(query_times)*1000,
        "P95Latency": percentile_latency(query_times,95)*1000,
        "QPS": 1/np.mean(query_times),
        "MemoryMB": memory_mb()
    }

# ====================== SQLite-VSS IVF =======================
def test_vss_ivf():
    if not FAISS_AVAILABLE:
        return None
    n_clusters = 24
    start_build = time.perf_counter()
    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, n_clusters)
    index.train(xb_vec)
    index.add(xb_vec)
    index.nprobe = 6
    build_time = time.perf_counter()-start_build

    query_times=[]
    for q in xq_vec:
        start = time.perf_counter()
        D,I = index.search(q.reshape(1,-1), TOPK)
        query_times.append(time.perf_counter()-start)
    return {
        "BuildTime": build_time,
        "AvgLatency": np.mean(query_times)*1000,
        "P95Latency": percentile_latency(query_times,95)*1000,
        "QPS": 1/np.mean(query_times),
        "MemoryMB": memory_mb()
    }

# ====================== Run Tests =======================
systems = {
    "SQLite-Brute": test_sqlite,
    "Hybrid-KDTree": test_hybrid
}
if FAISS_AVAILABLE:
    systems["SQLite-VSS-Flat"] = test_vss_flat
    systems["SQLite-VSS-IVF"] = test_vss_ivf

results={}
for name, func in systems.items():
    res = func()
    if res:
        results[name]=res

df = pd.DataFrame(results).T

# ====================== Print Table =======================
print("\n📊 DATABASE PERFORMANCE COMPARISON (30K vectors)")
print(df.round(4).to_string())