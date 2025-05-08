from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from .cfg import CFG, ROOT

def get_model(name: str):
    info = CFG.model[name]
    local = ROOT / info['local']
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists
        snapshot_download(info['repo'],
                          local_dir=str(local), # snapshot_download expects string path
                          resume_download=True,
                          local_dir_use_symlinks=False,
                          endpoint=CFG.model.get('hf_mirror')) # Use .get for optional hf_mirror
    return SentenceTransformer(str(local),
                               device=CFG.device,
                               trust_remote_code=True) 