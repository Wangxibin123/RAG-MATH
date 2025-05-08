# ---------- BEGIN Dockerfile ----------
# 1) 基础：轻量 micromamba；如没有 GPU 换 cpu 基镜像也可
FROM mambaorg/micromamba:1.5.7-cuda-12.2-debian-11

# 2) 复制项目源码
WORKDIR /app
COPY . /app

# 3) 安装依赖
RUN micromamba install -y -n base -c conda-forge \
      python=3.11 pytorch=2.2 sentence-transformers=2.7 transformers=4.40 \
      pandas=2.2 openpyxl=3.1 pymilvus=2.4 regex \
      huggingface-hub=0.23 uvicorn fastapi tqdm && \
    micromamba clean -a -y

# 4) (可选) 构建时预下载模型，减少冷启动时间
RUN python - <<'PY'
import yaml, subprocess, pathlib, os
cfg = yaml.safe_load(open("conf/model.yaml"))
for k,v in cfg.items():
    if k=="hf_mirror":continue
    p=pathlib.Path(v["local"])
    if not p.exists():
        subprocess.run(["python","-m","huggingface_hub.snapshot_download",
                        v["repo"],"--local-dir",str(p),
                        "--local-dir-use-symlinks","False","--resume-download","True"],
                       check=True)
PY

# 5) 入口
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn","gaokao_rag.api:app","--host","0.0.0.0","--port","8000"]
# ----------  END  Dockerfile ----------