# WHU ZXR - SQLite Enhanced Research Project

本项目是一个基于 SQLite 的深度研究与实验平台，旨在探索 SQLite 在**向量检索（Vector Search）**和**图计算（Graph Computing）**领域的扩展能力与性能优化。项目包含了 SQLite 核心源码、自定义扩展插件（sqlite-vec, sqlite-graph）以及一系列用于性能评估和算法验证的实验脚本。

## 📂 项目结构

```text
.
├── experiment/             # 算法实验与性能测试脚本
│   ├── test_v3.py          # KDTree + IVF 混合索引查询实验
│   ├── test_v4.py          # 改进版索引算法实验
│   └── ...                 # 内存占用、构建时间等多维度对比脚本
├── sqlite/                 # SQLite 官方源码仓库 (v3.x)
│   ├── src/                # 核心 C 语言源码
│   ├── ext/                # 官方扩展 (FTS5, RTree, JSON 等)
│   └── build/              # 编译产物与配置
├── sqlite-vec/             # 向量搜索扩展插件
│   ├── sqlite-vec.c        # 核心实现：支持 float/half/binary 向量
│   ├── zxrplugin.c         # 自定义插件示例 (Hello World 函数)
│   └── examples/           # Python/C 调用示例
├── sqlite-graph/           # 图数据库扩展插件
│   ├── include/graph.h     # 图节点、边及算法接口定义
│   ├── src/                # Cypher 解析器与图算法实现
│   └── tests/              # 单元测试与 TCK 场景测试
├── testsql/                # 数据库性能对比测试
│   ├── comparison.py       # SQLite vs DuckDB 性能基准测试
│   └── testsql/            # SQL 测试用例集 (.sql)
└── README.md               # 项目说明文档