import pandas as pd
import numpy as np
import json
from sentence_transformers import CrossEncoder
from .hub import get_model
from .embed import encode
from .score import hybrid
from .cfg import CFG, ROOT
# Dynamic store import based on configuration
if CFG.store_name == "milvus":
    from .store import milvus as store_module
elif CFG.store_name == "faiss":
    from .store import faiss as store_module
else:
    raise ImportError(f"Unsupported store type: {CFG.store_name}. Check conf/base.yaml.")

# -------- data frame 缓存 --------
# Ensure the data path is robust
DATA_FILE_PATH = ROOT / 'data/df_gk_math.xlsx'
DF = None
def load_dataframe():
    global DF
    if not DATA_FILE_PATH.exists():
        print(f"Warning: Data file not found at {DATA_FILE_PATH}. Queries might fail or use empty data.")
        # Create an empty DataFrame with expected columns to prevent downstream errors
        DF = pd.DataFrame(columns=['id', 'stem', 'difficulty']).set_index("id")
        return
    try:
        DF = pd.read_excel(DATA_FILE_PATH).set_index("id")
    except Exception as e:
        print(f"Error loading data from {DATA_FILE_PATH}: {e}")
        DF = pd.DataFrame(columns=['id', 'stem', 'difficulty']).set_index("id")
    DF.index = DF.index.astype(str)             # 只需在 load_dataframe() 里做一次

load_dataframe() # Load on module import

# -------- store 选择 --------
if CFG.store_name == "milvus":
    STORE = store_module.MilvusStore()
elif CFG.store_name == "faiss":
    STORE = store_module.FaissStore() # Assuming FaissStore is defined in faiss.py

# -------- reranker --------
# Use a try-except block for model loading for robustness
CE = None
try:
    # Assuming get_model("rerank") returns a SentenceTransformer-like object
    # and CrossEncoder needs the model path or name from it.
    # The original code was CE = CrossEncoder(get_model("rerank").model)
    # This implies get_model("rerank") has a .model attribute which is the path/name for CrossEncoder
    rerank_model_obj = get_model("rerank")
    if hasattr(rerank_model_obj, 'model_path'): # Common for ST models
        reranker_model_name_or_path = str(rerank_model_obj.model_path)
    elif hasattr(rerank_model_obj, 'name_or_path'): # Common for HF models
        reranker_model_name_or_path = str(rerank_model_obj.name_or_path)
    else:
        # Fallback: if get_model returns the raw SentenceTransformer, its path is often the first arg
        reranker_model_name_or_path = str(ROOT / CFG.model["rerank"]["local"])

    CE = CrossEncoder(reranker_model_name_or_path, device=CFG.device)
except Exception as e:
    print(f"Error loading CrossEncoder model: {e}. Reranking might not work.")

def build_index():
    if DF is None or DF.empty:
        print("Error: DataFrame is not loaded or is empty. Cannot build index.")
        return
    ids, vecs = [], []
    print(f"Building index from {len(DF)} items...")
    for _id, row in DF.iterrows():
        if 'stem' not in row or pd.isna(row['stem']):
            print(f"Skipping item with id {_id} due to missing or NaN stem.")
            continue
        try:
            ids.append(str(_id)) # Ensure ID is string
            vecs.append(encode(str(row.stem)))
            
        except Exception as e:
            print(f"Error encoding item {_id} ('{row.stem}'): {e}")
    
    if not ids:
        print("No valid items to index after processing.")
        return

    #vecs_np = np.array(vecs, dtype=np.float32) #这里改了
    try:
        vecs_np = np.stack(vecs).astype('float32')   # (N, dim)
    except ValueError as e:
        # 堆叠失败 => 至少有一个子向量维度不同
        print("❌ np.stack failed:", e)
        for i, v in enumerate(vecs[:5]):            # 看前 5 条诊断
            print(f"vec[{i}].shape =", getattr(v, "shape", None))
        return
    
    STORE.build(ids, vecs_np) #这里改了
    print(f"Index built successfully with {len(ids)} items.")

def query(stem: str, k=None):
    if k is None:
        k = CFG.topk_return

    if DF is None:
        print("Error: DataFrame not loaded. Query cannot be processed.")
        return []
    if CE is None:
        print("Warning: CrossEncoder not loaded. Reranking will be skipped.")

    qv = encode(stem)
    cand_ids, _ = STORE.search(qv, CFG.topk_recall)

    if not cand_ids:
        return []

    # Filter out IDs not present in the DataFrame (if any inconsistencies)
    valid_cand_ids = [str(cid) for cid in cand_ids if cid in DF.index]
    if not valid_cand_ids:
        return []

    if CE is not None:
        cross_inp = [(stem, DF.loc[i, "stem"]) for i in valid_cand_ids]
        rerank_scores = CE.predict(cross_inp, convert_to_numpy=True)
    else: # Fallback if reranker is not available
        rerank_scores = [0.5] * len(valid_cand_ids) # Neutral score

    final_candidates = []
    for cid, rs_score in zip(valid_cand_ids, rerank_scores):
        difficulty = DF.loc[cid, "difficulty"] if "difficulty" in DF.columns and not pd.isna(DF.loc[cid, "difficulty"]) else None
        final_score = hybrid(rs_score, difficulty)
        final_candidates.append((cid, final_score))
    
    # Sort by final score descending and take top k
    final_sorted = sorted(final_candidates, key=lambda x: x[1], reverse=True)[:k]
    
    # Prepare results
    result_ids = [i for i, _ in final_sorted]
    if not result_ids:
        return []
        
    # Ensure all result_ids are in DF before .loc to prevent KeyError
    # This should be guaranteed by valid_cand_ids check, but as a safeguard:
    result_ids_in_df = [rid for rid in result_ids if rid in DF.index]
    if not result_ids_in_df:
        return []

    return json.loads(DF.loc[result_ids_in_df].to_json(orient="records", force_ascii=False)) 