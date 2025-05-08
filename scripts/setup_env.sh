# ---------- BEGIN scripts/setup_env.sh ----------
#!/usr/bin/env bash
set -euo pipefail

ENV=ragmath
PY=3.11

echo "🔧 installing mamba & conda‑pack ..."
conda install -n base -y -c conda-forge mamba conda-pack

echo "📦 create/update env $ENV ..."
if ! conda env list | grep -q "^$ENV\s"; then
  mamba create -n "$ENV" -y python="$PY"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "pip install -e ."
pip install -e .

command -v pytest &>/dev/null && pytest -q

echo "📦 conda-pack -> ${ENV}_env.tar.gz"
conda-pack -n "$ENV" -o "${ENV}_env.tar.gz" --force
echo "✅ done → ${ENV}_env.tar.gz"
# ----------  END  scripts/setup_env.sh ----------