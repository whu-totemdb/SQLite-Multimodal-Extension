# SQLite Multimodal Extension - 多模数据扩展研究
毕业设计：基于 SQLite 的嵌入式多模数据库扩展与性能优化

## 1. 项目背景

本项目作为毕业设计研究课题，基于原生 SQLite 内核进行二次开发，自研并集成向量检索扩展、图计算扩展，同时设计实现 **MRKD（KDTree + IVF）混合索引结构**，打造一套兼容标准 SQL、支持 **结构化数据 + 向量数据 + 图数据 + 文本数据** 的一体化嵌入式多模数据库实验平台。


## 2. 技术栈

| 类别 | 技术 / 组件 | 用途 |
|-------|------------|------|
| 底层内核 | C 语言 + SQLite 3.x 原生源码 | 数据库内核、虚拟表、扩展接口开发 |
| 扩展插件 | C 动态链接库（`.so`） | 向量检索扩展、图计算扩展 |
| 编译工具 | GCC、Make、Shell 脚本 | 源码编译、动态库构建、工程管理 |
| 测试实验 | Python 3 | 性能压测、数据生成、基准对比、自动化实验 |
| 查询语法 | 标准 SQL + 扩展语法 | 多模数据增删改查、向量 / 图专属查询 |


## 3. 推荐运行环境

- **操作系统：** Ubuntu 18.04 / 20.04 / 22.04（推荐），Windows WSL 可兼容；
- **基础依赖：** `gcc`、`make`、`tcl-dev`、`python3`、`python3-pip`；
- **权限要求：** 普通用户即可，编译动态库无需 root（系统目录除外）。


## 4. 项目目录结构

```text
SQLite-Multimodal-Extension/
├── README.md
├── mybuild.sh
├── sqlite/
│   ├── src/
│   ├── ext/
│   └── build/
├── sqlite-vec/
│   ├── sqlite-vec.c
│   ├── zxrplugin.c
│   └── examples/
├── sqlite-graph/
│   ├── include/
│   │   └── graph.h
│   ├── src/
│   └── tests/
├── experiment/
│   ├── test_v3.py
│   ├── test_v4.py
│   ├── test_memory.py
│   ├── test_build_time.py
│   └── ...
├── testsql/
│   ├── comparison.py
│   └── testsql/
├── init.sql
└── Makefile
```

### 目录说明

| 路径 | 功能 |
|------|------|
| `sqlite/` | SQLite 官方源码 |
| `sqlite-vec/` | 自研向量检索扩展 |
| `sqlite-graph/` | 自研图计算扩展 |
| `experiment/` | 性能实验与算法验证 |
| `testsql/` | SQLite 与 DuckDB 对比测试 |
| `init.sql` | 初始化脚本 |
| `mybuild.sh` | 一键编译脚本 |

## 5. 使用指南

### 5.1 安装系统依赖

```bash
sudo apt update
sudo apt install gcc make tcl-dev
```

### 5.2 编译 SQLite 原生内核

#### 解压源码并创建编译目录

```bash
tar xzf sqlite.tar.gz

mkdir bld
cd bld

../sqlite/configure
```

#### 编译核心程序

```bash
make sqlite3

make sqlite3.c

make sqldiff
```

编译完成后，`bld/` 目录下会生成 `sqlite3` 可执行文件。

### 5.3 编译多模扩展插件

#### 方式一：一键编译（推荐）

```bash
chmod +x mybuild.sh

./mybuild.sh
```

#### 方式二：手动编译

##### 编译向量扩展

```bash
gcc -g -fPIC -shared sqlite-vec/sqlite-vec.c \
    -o vec0.so \
    -I./sqlite/src
```

#### 编译图扩展

```bash
gcc -g -fPIC -shared sqlite-graph/src/graph.c \
    -o graph0.so \
    -I./sqlite/src \
    -I./sqlite-graph/include
```

参数说明：

- `-fPIC`：生成位置无关代码；
- `-shared`：编译动态库；
- `-I`：指定头文件路径。

编译成功后将生成：

```text
vec0.so
graph0.so
```

### 5.4 加载扩展

启动 SQLite：

```bash
cd bld

./sqlite3
```

加载扩展：

```sql
.load vec0.so

.load graph0.so
```

若提示找不到文件：

```sql
.load /home/xxx/project/vec0.so
```


### 5.5 创建多模数据表

```sql
CREATE VIRTUAL TABLE mytable USING vec0(
    id INTEGER,
    news TEXT,
    embedding FLOAT[8],
    create_index = hybrid
);
```

#### 字段说明

| 字段 | 说明 |
|------|------|
| `id` | 主键 ID |
| `news` | 文本字段 |
| `embedding` | 定长向量 |
| `create_index` | 索引类型 |

### 5.6 数据操作示例

#### 插入数据

```sql
INSERT INTO mytable(id, news, embedding)
VALUES (
    1,
    '测试新闻数据',
    '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'
);
```

---

#### 向量相似检索

```sql
SELECT id, news, embedding
FROM mytable
WHERE embedding NEAR
'[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9]'
LIMIT 5;
```

### 5.7 执行初始化脚本

```sql
.read init.sql
```

## 常见问题

## 问题 1：fPIC not allowed

### 原因

缺少：

```text
-fPIC
```

### 解决方案

重新编译：

```bash
gcc -fPIC ...
```

## 问题 2：扩展加载失败

### 原因

- 路径错误；
- 架构不一致。

### 解决方案

```sql
.load /absolute/path/vec0.so
```

确认：

```text
Linux x86_64
```

环境一致。

## 问题 3：ModuleNotFoundError

安装依赖：

```bash
pip3 install numpy pandas
```


## 问题 4：tcl.h not found

安装：

```bash
sudo apt install tcl-dev
```

## 问题 5：混合索引创建失败

检查：

- `vec0.so` 是否成功加载；
- SQLite 与扩展是否使用相同版本源码编译。


