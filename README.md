# Gaokao-RAG

> Retrieval-Augmented Generation for Chinese *Gaokao* math problems  
> 双塔召回 ✚ Milvus / Faiss ✚ Cross-Encoder 精排 ✚ FastAPI REST

---

## 0 Directory layout *(v 1.2)*

RAG_MATH/
├─ conf/                 # YAML config (hot-reload)
│  ├─ base.yaml          # device / store / top-k / …
│  ├─ model.yaml         # HF repo ↔ local cache
│  ├─ milvus.yaml ─┐     # back-end specific
│  └─ faiss.yaml  ─┘
├─ data/df_gk_math.xlsx  # raw Gaokao questions
├─ models/               # HF weights & faiss_index*.bin
├─ gaokao_rag/           # Python package
│  ├─ cfg.py       # load conf/*
│  ├─ hub.py       # local-first model loader
│  ├─ embed.py     # text+LaTeX → 1664-d vec
│  ├─ text_only.py # text-only → 896-d vec
│  ├─ store/       # milvus.py / faiss.py
│  ├─ retriever.py # recall + rerank
│  ├─ api.py       # FastAPI server
│  └─ cli.py       # ragmath 
├─ scripts/
│  ├─ setup_env.sh       # conda-pack helper
│  └─ docker-compose.yml
└─ Dockerfile

---

## 1 Quick start (local CPU / GPU)

```bash
# clone & enter
git clone https://github.com/<you>/gaokao_rag.git
cd gaokao_rag

# env  (≈5 min with CN mirror)
bash scripts/setup_env.sh
conda activate ragmath

# build two indices
ragmath import          # mixed 1664-d  -> models/faiss_index.bin
ragmath import-text     # text  896-d  -> models/faiss_text.bin

# query
ragmath query       "∫_0^1 x dx"      -k 5   # mixed
ragmath query-text "设抛物线 y^2=2px" -k 5   # text-only

Milvus back-end – set store: milvus in conf/base.yaml, indices
will be stored in collections gaokao_math_emb and gaokao_math_emb_text.

⸻

2 Docker + FastAPI

docker compose -f scripts/docker-compose.yml build rag   # build image
docker compose -f scripts/docker-compose.yml up -d       # milvus + rag

# once per data update
curl -X POST http://localhost:8000/import

# health & stream query
curl http://localhost:8000/health
curl -N -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"stem":"设抛物线 y^2=2px 经过点 (2,2) 求 p"}'

path	method	body	note
/health	GET	–	ping
/import	POST	–	rebuild both indices
/query	POST	{"stem":"…","topk":10}	mixed 1664 • JSON-Lines
/query_text	POST	same	text 896 • JSON-Lines



⸻

3 Updating data
	1.	Append new rows to data/df_gk_math.xlsx (id, stem required).
	2.	Run

ragmath import
ragmath import-text

Any DataFrame‐loadable format works (CSV, Parquet, DB query) as long as
columns are the same.

⸻

4 Config cheatsheet (conf/base.yaml)

key	meaning
device	cuda:0, cpu, …
topk.recall / return	ANN size → final size (30 / 10)
difficulty.coeff	blend semantic vs. difficulty (0–1)
store	faiss (default) • milvus

Edit → restart – no code change needed.

⸻

5 CLI commands

command	description
ragmath import	build / update mixed index
ragmath import-text	build / update text index
ragmath query "<q>"     -k 5	mixed search
ragmath query-text "<q>" -k 5	text search
ragmath dump / load	save / restore FAISS mixed index



⸻

6 Architecture (one glance)

Excel → encode( 768 + 896 = 1664 ) → ANN(HNSW)─30 ─┐
                                                   │
                 KaLM-embed (TXT) ─┐               ▼
 LaTeX → MathBERTa (MATH) ─┐       └─ Cross-Encoder rerank ─► Top-k JSON



⸻

7 Dev tips
	•	Change models → edit conf/model.yaml; embed_dim auto-detects.
	•	Add back-end → implement store/base.py interface.
	•	CI example in .github/workflows/test.yml (pytest + smoke query).

⸻

License

Apache-2.0 © 2025 XIBIN.
Bundled model weights retain their original licenses (see models/).

特别注意models因为过大被放到了gaokao_rag之外