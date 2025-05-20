# Gaokao-RAG

> Retrieval-Augmented Generation for Chinese *Gaokao* (College Entrance Exam) math problems.  
> 本项目旨在为高考数学题目提供基于检索增强生成的解决方案，结合双塔召回、向量存储 (Milvus/Faiss)、Cross-Encoder 精排及 FastAPI 服务。

---

## ✨ 功能特性

*   **混合检索**: 同时理解题目文本和 LaTeX 公式，生成高质量的向量表示。
*   **高效索引**: 使用 Faiss 或 Milvus 进行大规模向量索引和快速相似性搜索。
*   **精准重排**: 通过 Cross-Encoder 模型对召回结果进行精排，提升匹配精度。
*   **灵活配置**: 通过 YAML 文件管理模型路径、服务参数等，支持热重载。
*   **便捷 API**: 提供 FastAPI 接口，方便与前端或其他服务集成。
*   **命令行工具**: `ragmath` CLI 支持索引构建、查询等操作。
*   **Docker 支持**: 提供 Dockerfile 和 docker-compose 配置，方便部署。

---

## 📚 项目结构 (Project Structure)

```
RAG_MATH/
├── .git/                     # Git 版本控制目录
├── .github/                  # GitHub 相关配置
│   └── workflows/
│       └── test.yml          # GitHub Actions CI/CD 配置文件 (示例)
├── conf/                     # YAML 配置目录 (支持热重载)
│   ├── base.yaml             # 基础配置: 设备, 存储后端, top-k 参数等
│   ├── model.yaml            # 模型配置: HuggingFace 仓库名 ↔ 本地模型路径
│   ├── milvus.yaml           # Milvus 特定配置
│   └── faiss.yaml            # Faiss 特定配置
├── data/                     # 数据文件目录
│   ├── df_gk_math.xlsx       # 核心数据：高考数学题目（Excel 格式）
│   ├── .~df_gk_math.xlsx     # Excel 临时文件 (已被 .gitignore 忽略)
│   └── df_gk_math_副本.xlsx  # 数据副本示例 (请确保只使用一个主数据文件)
├── dist/                     # Python 包构建输出目录 (例如 .whl, .tar.gz)
├── gaokao_rag/               # Python 主程序包
│   ├── __init__.py           # 包初始化文件
│   ├── api.py                # FastAPI 服务端接口定义
│   ├── cfg.py                # 加载 conf/* 配置文件的模块
│   ├── cli.py                # ragmath 命令行工具入口
│   ├── embed.py              # 文本 + LaTeX → 混合向量编码逻辑
│   ├── formula.py            # LaTeX 公式处理相关 (如果独立)
│   ├── hub.py                # 本地优先的模型加载器
│   ├── retriever.py          # 核心检索逻辑：召回 + 重排
│   ├── score.py              # 混合打分逻辑 (例如结合相似度与难度)
│   ├── text_only.py          # 纯文本 → 文本向量编码逻辑
│   ├── store/                # 向量存储后端实现
│   │   ├── __init__.py
│   │   ├── base.py           # 存储后端基类接口
│   │   ├── faiss.py          # Faiss 后端实现
│   │   └── milvus.py         # Milvus 后端实现
│   ├── __pycache__/          # Python 编译缓存 (已被 .gitignore 忽略)
│   ├── conf/                 # (此为旧版结构，配置文件已移至项目根目录的 conf/)
│   └── data/                 # (此为旧版结构，数据文件已移至项目根目录的 data/)
├── models/                   # 模型权重和 Faiss 索引存储目录
│   └── .gitkeep              # 用于保持 models 目录结构 (此目录内容通过 .gitignore 排除)
├── scripts/                  # 辅助脚本目录
│   ├── setup_env.sh          # Conda 环境一键设置脚本
│   └── docker-compose.yml    # Docker Compose 配置文件 (用于 Milvus 和 RAG 服务)
├── .DS_Store                 # macOS 文件夹元数据 (已被 .gitignore 忽略)
├── .gitignore                # 指定 Git 应忽略的文件和目录
├── Dockerfile                # Docker 镜像构建文件
├── LICENSE                   # 项目许可证文件
├── pyproject.toml            # Python 项目配置文件 (PEP 621, for poetry/setuptools)
└── README.md                 # 您正在阅读的这个文件
```
**注意**:
*   `models/` 目录下的模型文件和生成的 Faiss 索引文件由于体积较大，已被添加到 `.gitignore` 中，不会提交到 Git 仓库。您需要根据 `conf/model.yaml` 的配置，自行下载或准备这些模型文件到本地的 `models/` 目录。
*   配置文件已从 `gaokao_rag/conf/` 移至项目根目录的 `conf/`。
*   数据文件已从 `gaokao_rag/data/` 移至项目根目录的 `data/`。

---

## 🚀 快速开始与部署

### 1. 环境准备

**a. 克隆项目**
   ```bash
   git clone https://github.com/Wangxibin123/RAG-MATH.git # 请替换为您的仓库地址
   cd RAG-MATH
   ```

**b. 创建 Conda 环境**
   项目提供了一个便捷的脚本来创建和配置 Conda 环境 (包含所有依赖)。
   ```bash
   bash scripts/setup_env.sh
   conda activate ragmath
   ```
   这大约需要5分钟（取决于您的网络和是否使用国内镜像）。

**c. 准备模型文件**
   *   根据 `conf/model.yaml` 中列出的模型名称 (例如 `text_model`, `math_model`, `rerank`)，从 Hugging Face Hub 或其他来源下载相应的预训练模型。
   *   将下载的模型文件存放到项目根目录下的 `models/` 目录中，并确保其子目录结构与 `conf/model.yaml` 中 `local` 字段指定的路径匹配。
   *   **示例**: 如果 `model.yaml` 中有：
     ```yaml
     text_model:
       hf: BAAI/bge-m3 # 假设这是HF上的模型名
       local: bge-m3    # 对应 models/bge-m3/ 目录
     ```
     您需要将 `BAAI/bge-m3` 模型下载到 `RAG_MATH/models/bge-m3/` 目录下。

**d. 准备数据文件**
   *   核心数据文件是 `data/df_gk_math.xlsx`。请确保此文件存在，并且包含至少 `id` (唯一标识) 和 `stem` (题干) 两列。其他可选列如 `difficulty`, `answer`, `source` 等可以根据您的需求添加。

**e. 构建向量索引**
   在提供 API 服务之前，您需要为您的题目数据构建向量索引。
   *   **默认使用 Faiss**:
     ```bash
     ragmath import          # 构建混合内容索引 (文本+公式) -> models/faiss_index.bin
     ragmath import-text     # 构建纯文本内容索引 -> models/faiss_text.bin
     ```
     索引文件将保存在 `models/` 目录下。
   *   **如果使用 Milvus**:
     1.  确保 Milvus 服务已启动 (可以使用 `scripts/docker-compose.yml` 来启动 Milvus 实例)。
     2.  在 `conf/base.yaml` 中设置 `store_name: milvus`。
     3.  然后运行相同的导入命令，数据将被存入 Milvus 集合中。

### 2. 启动 FastAPI 后端服务

使用 Uvicorn 启动 FastAPI 应用：
```bash
uvicorn gaokao_rag.api:app --reload --host 0.0.0.0 --port 8000
```
*   `--reload`: 当代码更改时自动重启服务，方便开发。生产环境可以去掉。
*   `--host 0.0.0.0`: 使服务可以从外部网络访问（例如在 Docker 容器内或局域网其他机器访问）。如果只在本地访问，可以使用默认的 `127.0.0.1`。
*   `--port 8000`: 指定服务监听的端口。

服务启动成功后，您会看到类似以下的输出：
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using statreload
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

---

## 🤖 API 端点详解

API 服务提供了以下主要端点：

### `/health`

*   **方法**: `GET`
*   **描述**: 检查服务的健康状态。
*   **请求体**: 无
*   **成功响应 (200 OK)**:
    ```json
    {
        "status": "ok",
        "ts": 1678886400.123456 // 当前服务器时间戳
    }
    ```

### `/import`

*   **方法**: `POST`
*   **描述**: 触发后台任务，重新构建向量索引。当您更新了 `data/df_gk_math.xlsx` 文件后，应调用此接口。
*   **请求体**: 无
*   **成功响应 (202 Accepted)**:
    ```json
    {
        "msg": "import started"
    }
    ```
    注意：这只表示任务已开始，实际索引构建可能需要一些时间。

### `/query` (流式，旧版查询接口)

*   **方法**: `POST`
*   **描述**: 执行混合内容（文本+公式）检索和重排，以流式 JSON Lines 格式返回结果。
*   **请求体**:
    ```json
    {
        "stem": "你的查询题目文本，可以包含 LaTeX",
        "topk": 10 // 可选，期望返回的结果数量，默认为 conf/base.yaml 中的设置
    }
    ```
*   **成功响应 (200 OK)**: `Content-Type: application/json` (实际上是 JSON Lines)
    每一行是一个 JSON 对象，代表一个匹配的题目，包含原始 DataFrame 中的所有字段以及可能的得分（具体字段取决于 `retriever.query` 旧版的实现）。

### `/query_text` (流式，旧版纯文本查询接口)

*   与 `/query` 类似，但通常用于纯文本的检索流程。

---

### ⭐ `/api/v1/match_problems` (推荐使用的新版题目匹配接口)

*   **方法**: `POST`
*   **描述**: 根据输入的题目信息（题干），从题库中检索并返回最相似的 K 道题目，每道题目包含其 ID、题干和相似度得分。这是为前端或外部服务设计的、更简洁的题目匹配接口。
*   **请求体 (Request Body)**: `Content-Type: application/json`
    ```json
    {
        "query_stem": "这里是您要查询的题目文本，可以包含 LaTeX 公式，例如：已知函数 f(x) = x^2 + 2x - 3，求其对称轴。",
        "top_k": 5
    }
    ```
    *   `query_stem` (string, **必需**): 您希望用来匹配的题目内容。
    *   `top_k` (integer, 可选, 默认值: 5): 您希望返回的最相似题目的数量。有效范围通常在 1 到 50 之间（具体可由 Pydantic 模型定义）。

*   **成功响应 (200 OK)**: `Content-Type: application/json`
    ```json
    {
        "matched_problems": [
            {
                "id": "retrieved_problem_id_1",
                "stem": "这是匹配到的第一道题目的题干内容...",
                "score": 0.9523
            },
            {
                "id": "retrieved_problem_id_2",
                "stem": "这是匹配到的第二道题目的题干内容...",
                "score": 0.8871
            }
            // ... 最多 K 项
        ]
    }
    ```
    *   `matched_problems` (array of objects): 一个包含匹配到的题目的列表。
        *   `id` (string): 匹配到的题目的唯一 ID。
        *   `stem` (string): 匹配到的题目的完整题干。
        *   `score` (float): 该题目与输入查询之间的相似度得分。分数越高，表示越相似。这个得分通常是由重排器给出或者混合了多种因素的最终得分。

*   **错误响应**:
    *   `422 Unprocessable Entity`: 请求体验证失败（例如，`query_stem` 缺失，`top_k` 类型错误或超出范围）。响应体会包含详细的错误信息。
        ```json
        {
            "detail": [
                {
                    "loc": ["body", "query_stem"],
                    "msg": "field required",
                    "type": "value_error.missing"
                }
            ]
        }
        ```
    *   `500 Internal Server Error`: 服务器内部在处理请求时发生错误（例如，模型加载失败，检索过程中出现意外异常等）。
        ```json
        {
            "detail": "Internal server error while matching problems."
        }
        ```

---

## 🛠️ 命令行工具 (`ragmath`)

除了 FastAPI 服务，项目还提供了一个命令行工具 `ragmath` 用于常见操作：

| 命令                             | 描述                                         |
|----------------------------------|----------------------------------------------|
| `ragmath import`                 | 构建/更新混合内容索引 (文本+公式)            |
| `ragmath import-text`            | 构建/更新纯文本内容索引                      |
| `ragmath query "<query_stem>"`   | 执行混合内容查询 (旧版，直接输出到终端)      |
| `ragmath query-text "<query_stem>"`| 执行纯文本内容查询 (旧版，直接输出到终端)    |
| `ragmath dump`                   | 保存 Faiss 混合内容索引到文件 (如果使用 Faiss) |
| `ragmath load`                   | 从文件加载 Faiss 混合内容索引 (如果使用 Faiss) |

> 使用 `-k <number>` 参数可以为 `query` 和 `query-text` 命令指定返回结果的数量。

---

## ⚙️ 配置 (`conf/` 目录)

所有重要的配置都在项目根目录的 `conf/` 目录下的 YAML 文件中：

*   **`base.yaml`**: 核心配置，如：
    *   `device`: 计算设备 (`cpu`, `cuda:0` 等)。
    *   `store_name`: 使用的向量存储后端 (`faiss` 或 `milvus`)。
    *   `topk_recall`: ANN 初步召回的数量。
    *   `topk_return`: 经过重排后最终返回给旧版 `/query` 接口的数量。
    *   `difficulty_coeff`: 语义相似度与题目难度融合系数 (0–1)。
*   **`model.yaml`**: 定义了项目中用到的各种模型 (文本嵌入、数学公式嵌入、重排器) 的 Hugging Face Hub名称及其对应的本地存储路径 (相对于 `models/` 目录)。
*   **`faiss.yaml`**: Faiss 特定的配置，例如索引文件的前缀。
*   **`milvus.yaml`**: Milvus 特定的配置，例如连接参数、集合名称。

> 大部分配置项修改后，如果 FastAPI 服务以 `--reload` 模式启动，会自动重载。

---

## 🧩 架构概览

```mermaid
graph LR
    A[用户/前端] -->|1. API 请求 (题目, top_k)| FASTAPI{FastAPI 服务<br>(api.py)};
    FASTAPI -->|2. 调用检索逻辑| RETRIEVER[检索模块<br>(retriever.py)];
    
    subgraph "检索模块内部 (retriever.query)"
        RETRIEVER -->|3. 编码查询题目| EMBED[嵌入模块<br>(embed.py)];
        EMBED --> QV[查询向量];
        QV -->|4. ANN搜索 Top-N'| STORE[向量存储<br>(Faiss/Milvus)];
        STORE -->|5. 候选ID列表| CAND_IDS;
        CAND_IDS -->|6. 获取候选题目文本| DATAFRAME[DataFrame (题库)];
        DATAFRAME -->|7. (查询题目, 候选题目)| RERANKER[Cross-Encoder<br>重排器];
        RERANKER -->|8. 重排后得分| RS_SCORES;
        RS_SCORES -->|9. 结合难度等因素| HYBRID_SCORE[混合打分<br>(score.py)];
        HYBRID_SCORE -->|10. 最终排序与筛选 Top-K| FINAL_CANDS[最终候选(ID, Stem, Score)];
    end

    FINAL_CANDS -->|11. 格式化结果| FASTAPI;
    FASTAPI -->|12. JSON 响应| A;

    subgraph "模型与数据"
      MODELS_CONF[conf/model.yaml] -->|定义模型路径| EMBED;
      MODELS_CONF --> RERANKER;
      EXCEL_DATA[data/df_gk_math.xlsx] -->|构建索引/提供文本| STORE;
      EXCEL_DATA --> DATAFRAME;
    end
```

---

## 🔧 开发与扩展提示

*   **更改模型**: 编辑 `conf/model.yaml` 指定新的模型名称和本地路径。`embed_dim` 等参数通常会自动检测，但请确保新模型的输出与现有流程兼容。
*   **添加存储后端**: 如果要支持新的向量数据库，需在 `gaokao_rag/store/` 目录下创建新的实现，并继承 `gaokao_rag/store/base.py` 中的接口。然后在 `conf/base.yaml` 中引用新的 `store_name`，并更新 `gaokao_rag/retriever.py` 中的动态导入逻辑。
*   **CI/CD**: 项目包含一个 `.github/workflows/test.yml` 示例，展示了如何使用 GitHub Actions 进行基本的 `pytest` 测试和冒烟查询。您可以根据需要扩展。

---

## 📜 许可证 (License)

Apache-2.0 © 2024 XIBIN (请替换为实际年份和您的名字/组织)

> 本项目中使用的预训练模型权重保留其原始许可证。相关模型的许可证信息通常可以在其 Hugging Face 仓库或源项目中找到。
> **重要**: `models/` 目录中的模型文件由于体积较大，已通过 `.gitignore` 文件将其排除在 Git 版本控制之外。请在本地根据 `conf/model.yaml` 的指引准备所需的模型文件。