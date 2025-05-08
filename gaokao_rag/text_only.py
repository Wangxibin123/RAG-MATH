# gaokao_rag/text_only.py
from pathlib import Path
import numpy as np, pandas as pd, json
from tqdm import tqdm
from .cfg import CFG, ROOT
from .hub import get_model
from .formula import split

# ---------------- 数据和模型 ----------------
DATA_FILE = ROOT / "data/df_gk_math.xlsx"
DF = pd.read_excel(DATA_FILE).set_index("id")
DF.index = DF.index.astype(str)

_text = get_model("text")
TEXT_DIM = _text.get_sentence_embedding_dimension()

def encode_text_only(txt: str):
    rep, _ = split(txt)                 # 去掉公式占位符
    return _text.encode(rep, normalize_embeddings=True).astype('float32')

# ---------------- 选后端 ----------------
if CFG.store_name == "milvus":
    from .store import milvus as store_module
    class MilvusText(store_module.MilvusStore):
        def __init__(self):
            super().__init__()
            self.dimension = TEXT_DIM            # ★ 改维度
        def _create(self, p):
            p = dict(p)
            p["collection"] += "_text"  # 独立集合
            return super()._create(p)
    STORE = MilvusText()
else:
    # text_only.py 选后端里的 else 分支
    from .store import faiss as store_module
    class FaissText(store_module.FaissStore):
        """Text-only (896-dim) FAISS store — 独立文件 faiss_text.bin"""
        def __init__(self):
            
            # ① 构造一个专属的 param，指向 faiss_text.bin
            path = ROOT / "models/faiss_text.bin"
            super().__init__(index_path_override=str(path), dimension_override=TEXT_DIM)


    STORE = FaissText()

# ---------------- 构建索引 ----------------
def build_text_index():
    ids, vecs = [], []
    for _id, row in tqdm(DF.iterrows(), total=len(DF), desc="encode(text)"):
        if pd.isna(row.stem): continue
        ids.append(str(_id))
        vecs.append(encode_text_only(str(row.stem)))
    STORE.build(ids, np.stack(vecs).astype('float32'))
    print(f"[text-only] index built: {len(ids)} vectors, dim={TEXT_DIM}")

# ---------------- 查询 --------------------
def query_text_only(stem: str, k: int = 10):
    qv = encode_text_only(stem)
    cand_ids, _ = STORE.search(qv, k)
    if not cand_ids: return []
    return json.loads(DF.loc[cand_ids].to_json(orient="records", force_ascii=False))