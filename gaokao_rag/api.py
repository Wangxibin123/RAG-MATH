# ---------- BEGIN gaokao_rag/api.py ----------
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import json, time
from .retriever import build_index, query

app = FastAPI(title="Gaokao-RAG")

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
# ----------  END  gaokao_rag/api.py ----------