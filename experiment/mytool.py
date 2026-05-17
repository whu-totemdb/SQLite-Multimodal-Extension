import json
from huggingface_hub import hf_hub_download
import numpy as np


def read_fvecs(filename):
    data = np.fromfile(filename, dtype=np.float32)
    dim = data.view(np.int32)[0]
    return data.reshape(-1, dim + 1)[:, 1:]


def load_dataset(dataset_type="fvecs"):
    """
    dataset_type:
        "fvecs"  -> 旧版 sift 数据
        "npy"    -> 新数据集格式
    """

    if dataset_type == "fvecs":
        print("Loading fvecs dataset...")

        vector_data = read_fvecs("hub/vectors.fvecs")
        query_vectors = read_fvecs("hub/query.fvecs")

    elif dataset_type == "npy":
        print("Loading npy dataset...")

        # 1️⃣ 加载向量
        vector_data = np.load("random_float_1m/vectors.npy")

        # 2️⃣ 加载 query（只提取 query 字段）
        query_vectors = []
        with open("random_float_1m/tests.jsonl", "r") as f:
            for line in f:
                obj = json.loads(line)
                query_vectors.append(obj["query"])

        query_vectors = np.array(query_vectors, dtype=np.float32)

    else:
        raise ValueError("Unknown dataset_type")

    return vector_data, query_vectors


def load_datasets(dataset_type, base_path):
    """
    dataset_type:
        "fvecs" -> 读取 sift fvecs 数据
        "npy"   -> 读取 vectors.npy + tests.jsonl

    base_path:
        数据集所在文件夹路径
    """

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

    else:
        raise ValueError("Unsupported dataset type")

    return vector_data.astype(np.float32), query_vectors.astype(np.float32)
