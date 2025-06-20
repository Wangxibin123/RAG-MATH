# ---------- BEGIN gaokao_rag/api.py ----------
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json, time
from typing import List, Dict, Any
from .retriever import build_index, query

# --- Pydantic Models for the new API ---
class MatchRequest(BaseModel):
    query_stem: str = Field(..., description="需要匹配的题目文本，可以包含 LaTeX 公式")
    top_k: int = Field(default=5, ge=1, le=50, description="期望返回的最相似题目的数量")

class MatchedProblem(BaseModel):
    id: str = Field(..., description="匹配到的题目的唯一ID")
    stem: str = Field(..., description="匹配到的题目的题干文本")
    score: float = Field(..., description="与输入题目的相似度得分")

class MatchResponse(BaseModel):
    matched_problems: List[MatchedProblem] = Field(..., description="匹配到的题目列表")
# --- End Pydantic Models ---

app = FastAPI(title="Gaokao-RAG")

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# --- health ---
@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}

# --- import (后台任务) ---
@app.post("/import")
def import_data(bg: BackgroundTasks):
    bg.add_task(build_index)
    return JSONResponse({"msg": "import started"}, status_code=202)

# --- query ---
class Q(BaseModel):
    stem: str
    topk: int | None = None

def _as_stream(items):
    for rec in items:
        yield json.dumps(rec, ensure_ascii=False) + "\n"
        time.sleep(0)

@app.post("/query")
async def api_query(q: Q):
    data = query(q.stem, q.topk)
    return StreamingResponse(_as_stream(data),
                             media_type="application/json")

# --- New API Endpoint: Match Problems ---
@app.post("/api/v1/match_problems", response_model=MatchResponse, tags=["Problem Matching"])
async def match_similar_problems(request: MatchRequest):
    """
    根据输入的题目信息，匹配并返回最相似的 K 道题目。
    """
    try:
        retrieved_items = query(request.query_stem, request.top_k)

        matched_problems_list: List[MatchedProblem] = []
        # 假设 retrieved_items 是一个可迭代对象，每个 item 是一个字典
        # 例如: {'id': 'some_id', 'score': 0.95, 'stem': '题干内容', 'source': ...}
        
        # 临时记录第一个item的结构，便于调试（如果需要）
        # first_item_for_debug = next(iter(retrieved_items), None)
        # if first_item_for_debug:
        #     print(f"DEBUG: First item from query(): {first_item_for_debug}")
        # # 注意：如果retrieved_items是生成器，上面这行会消耗第一个元素。
        # # 如果需要重新迭代，可能需要再次调用 query() 或将其转为list。
        # # 为安全起见，实际处理时，我们直接迭代。

        processed_items_count = 0
        for item in retrieved_items:
            processed_items_count += 1
            if not isinstance(item, dict):
                print(f"Skipping item, not a dictionary: {item}")
                continue

            problem_id = item.get('id')
            problem_score = item.get('score')
            problem_stem = item.get('stem')

            # 确保基本字段存在且类型可转换
            if problem_id is not None and problem_score is not None and problem_stem is not None:
                try:
                    matched_problems_list.append(
                        MatchedProblem(
                            id=str(problem_id),
                            stem=str(problem_stem),
                            score=float(problem_score)
                        )
                    )
                except ValueError as e:
                    print(f"Skipping item due to value conversion error: {item}, error: {e}")
            else:
                print(f"Skipping item due to missing id, score, or stem: {item}")
        
        # 调试信息：如果处理后列表为空但确实有检索到内容
        if not matched_problems_list and processed_items_count > 0:
            # 为了获取第一个元素用于调试而不影响主逻辑，可以再次调用 query
            # 或者在开发阶段将 retrieved_items 转为 list。
            # 这里我们只打印一个通用警告。
            print(f"Warning: No problems could be formatted. Processed {processed_items_count} items from query(). Check item structure and keys ('id', 'score', 'stem').")


        return MatchResponse(matched_problems=matched_problems_list)

    except Exception as e:
        print(f"Error during matching problems: {e}") # 临时打印
        # Consider logging the traceback for more detailed debugging
        # import traceback
        # print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error while matching problems.")
# ----------  END  gaokao_rag/api.py ----------