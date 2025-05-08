# Gaokao-RAG

> Retrieval-Augmented Generation for Chinese *Gaokao* math problems  
> 双塔召回 ✚ Milvus / Faiss ✚ Cross-Encoder 精排 ✚ FastAPI REST

---

## 0. 目录结构 (v1.2)

```
RAG_MATH/
├── conf/                       # YAML 配置 (支持热重载)
│   ├── base.yaml               # 设备 / 存储 / top-k 等全局配置
│   ├── model.yaml              # HuggingFace 仓库 ↔ 本地缓存模型路径配置
│   ├── milvus.yaml             # Milvus 特定配置
│   └── faiss.yaml              # Faiss 特定配置
├── data/
│   └── df_gk_math.xlsx         # 原始高考数学题目数据
├── models/                     # 模型权重和 Faiss 索引 (.gitkeep, 此目录内容不提交到 Git)
│   └── .gitkeep                # 用于保持 models 目录结构
├── gaokao_rag/                 # Python 主程序包
│   ├── cfg.py                  # 加载 conf/* 配置文件
│   ├── hub.py                  # 本地优先的模型加载器
│   ├── embed.py                # 文本 + LaTeX → 1664维 向量编码
│   ├── text_only.py            # 纯文本 → 896维 向量编码
│   ├── store/                  # 向量存储后端 (milvus.py / faiss.py)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── faiss.py
│   │   └── milvus.py
│   ├── retriever.py            # 召回 + 重排逻辑
│   ├── api.py                  # FastAPI 服务端接口
│   └── cli.py                  # ragmath 命令行工具入口
├── scripts/
│   ├── setup_env.sh            # Conda 环境设置脚本
│   └── docker-compose.yml      # Docker Compose 配置文件
├── .github/
│   └── workflows/
│       └── test.yml            # GitHub Actions CI/CD 配置文件
├── .gitignore                  # 指定 Git 忽略的文件和目录
├── Dockerfile                  # Docker 镜像构建文件
├── LICENSE                     # 项目许可证
├── pyproject.toml              # Python 项目配置文件 (PEP 621)
└── README.md                   # 就是您现在正在看的这个文件
```

---

## 1. 快速开始 (本地 CPU / GPU)

1.  **克隆仓库并进入目录**:
    ```bash
    git clone https://github.com/Wangxibin123/RAG-MATH.git # 请替换为您的仓库地址
    cd RAG-MATH
    ```

2.  **创建并激活 Conda 环境** (使用国内镜像大约需要5分钟):
    ```bash
    bash scripts/setup_env.sh
    conda activate ragmath
    ```

3.  **构建索引**:
    *   默认使用 Faiss。索引文件将保存在 `models/` 目录下 (例如 `models/faiss_index.bin` 和 `models/faiss_text.bin`)。
    ```bash
    ragmath import          # 构建混合内容索引 (文本+公式, 1664维)
    ragmath import-text     # 构建纯文本内容索引 (896维)
    ```
    *   **如果使用 Milvus**:
        *   请先确保 Milvus 服务已启动 (可参考 `scripts/docker-compose.yml`)。
        *   在 `conf/base.yaml` 中设置 `store: milvus`。
        *   索引将存储在 Milvus 的 `gaokao_math_emb` 和 `gaokao_math_emb_text` 集合中。

4.  **执行查询**:
    ```bash
    # 混合内容查询 (默认 k=5)
    ragmath query "∫_0^1 x dx" -k 5

    # 纯文本内容查询 (默认 k=5)
    ragmath query-text "设抛物线 y^2=2px" -k 5
    ```

---

## 2. Docker 与 FastAPI

1.  **构建 Docker 镜像**:
    ```bash
    docker compose -f scripts/docker-compose.yml build rag
    ```

2.  **启动服务 (Milvus + RAG API)**:
    ```bash
    docker compose -f scripts/docker-compose.yml up -d
    ```

3.  **导入数据 (首次运行或数据更新后执行)**:
    ```bash
    curl -X POST http://localhost:8000/import
    ```

4.  **API 端点示例**:

    *   **健康检查**:
        ```bash
        curl http://localhost:8000/health
        ```
    *   **流式查询 (混合)**:
        ```bash
        curl -N -X POST http://localhost:8000/query \\
             -H "Content-Type: application/json" \\
             -d \'{"stem":"设抛物线 y^2=2px 经过点 (2,2) 求 p"}\'
        ```

    **API 列表**:

    | 路径          | 方法 | 请求体                          | 备注                             |
    |---------------|------|-----------------------------------|----------------------------------|
    | `/health`     | GET  | –                                 | 服务健康状态                     |
    | `/import`     | POST | –                                 | 重建所有索引                     |
    | `/query`      | POST | `{"stem":"...", "topk":10}`       | 混合查询 (1664维), JSON Lines 输出 |
    | `/query_text` | POST | `{"stem":"...", "topk":10}`       | 纯文本查询 (896维), JSON Lines 输出 |

---

## 3. 更新数据

1.  向 `data/df_gk_math.xlsx` 文件中追加新的行 (至少需要 `id` 和 `stem` 列)。
2.  重新运行导入命令:
    ```bash
    ragmath import
    ragmath import-text
    ```
    或者，如果通过 Docker API 运行，则调用:
    ```bash
    curl -X POST http://localhost:8000/import
    ```
    > 支持任何可以被 Pandas DataFrame 加载的数据格式 (如 CSV, Parquet, 数据库查询结果等)，只要列名保持一致即可。

---

## 4. 配置速查 (`conf/base.yaml`)

| 键                 | 含义                                       |
|--------------------|--------------------------------------------|
| `device`           | 计算设备，例如 `cuda:0`, `cpu` 等            |
| `topk.recall`      | ANN 召回数量 (例如: 30)                    |
| `topk.return`      | 最终返回结果数量 (例如: 10)                |
| `difficulty.coeff` | 语义相似度与题目难度融合系数 (0–1)         |
| `store`            | 向量存储后端，可选 `faiss` (默认) 或 `milvus` |

> 修改配置文件后，服务会自动重载，无需重启。

---

## 5. 命令行工具 (`ragmath`)

| 命令                             | 描述                         |
|----------------------------------|------------------------------|
| `ragmath import`                 | 构建/更新混合内容索引        |
| `ragmath import-text`            | 构建/更新纯文本内容索引      |
| `ragmath query "<query_stem>"`   | 执行混合内容查询             |
| `ragmath query-text "<query_stem>"`| 执行纯文本内容查询           |
| `ragmath dump`                   | 保存 Faiss 混合内容索引到文件 |
| `ragmath load`                   | 从文件加载 Faiss 混合内容索引 |

> 使用 `-k <number>` 参数可以指定查询返回的结果数量。

---

## 6. 架构概览

```mermaid
graph LR
    A[Excel/数据源] --> B{编码模块};
    B -- 文本+公式 --> C[混合编码器 <br/> (768 + 896 = 1664维)];
    B -- 纯文本 --> D[文本编码器 <br/> (KaLM-embed, 896维)];
    C --> E[ANN索引 (HNSW)];
    D --> F[ANN索引 (HNSW, 文本专用)];

    subgraph "查询流程 (混合)"
        Q1[用户查询 (文本+公式)] --> B;
        E --> G[召回 Top-N];
    end

    subgraph "查询流程 (纯文本)"
        Q2[用户查询 (纯文本)] --> B;
        F --> H[召回 Top-N (文本专用)];
    end

    G --> I{Cross-Encoder 重排};
    H --> I;
    I --> J[Top-K 结果 (JSON)];

    subgraph "模型细节"
        M_TEXT[文本: KaLM-embed]
        M_MATH[公式: MathBERTa]
        M_TEXT --> C;
        M_MATH --> C;
    end
```

---

## 7. 开发提示

*   **更改模型**: 编辑 `conf/model.yaml`；`embed_dim` 会自动检测。
*   **添加存储后端**: 实现 `gaokao_rag/store/base.py` 中的接口。
*   **CI/CD**: 参考 `.github/workflows/test.yml` 中的示例 (包含 `pytest` 和冒烟测试)。

---

## 许可证 (License)

Apache-2.0 © 2024 XIBIN (请替换为实际年份和您的名字)

> 本项目中使用的预训练模型权重保留其原始许可证。相关模型的许可证信息通常可以在其 Hugging Face 仓库或源项目中找到。
> **注意**: 由于 `models/` 目录中的模型文件体积较大，已通过 `.gitignore` 文件将其排除在 Git 版本控制之外。请在本地准备所需的模型文件。