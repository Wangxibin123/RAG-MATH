import numpy as np
import torch # Ensure torch is imported if not already via other means
from .hub import get_model
# Assuming formula.py will be created correctly by the user later
# from .formula import split 
from .cfg import CFG

# Placeholder for split function if formula.py is problematic
# This is a fallback and should be replaced by importing from formula.py
def placeholder_split(text: str):
    print("Warning: Using placeholder_split. formula.py might not be loaded correctly.")
    return text, []

_text_model = get_model("text")
_math_model = get_model("math")

TEXT_DIM = _text_model.get_sentence_embedding_dimension()
MATH_DIM = _math_model.get_sentence_embedding_dimension()
AUTO_DIM = TEXT_DIM + MATH_DIM

if AUTO_DIM != CFG.embed_dim:
    print(f"[WARN] detected embed_dim={AUTO_DIM}, "
          f"override conf/base.yaml embed_dim={CFG.embed_dim}")
    CFG.embed_dim = AUTO_DIM     # 动态覆盖配置

def encode(sentence: str):
    # Try to import the real split, fallback to placeholder if it fails or not available
    try:
        from .formula import split
    except ImportError:
        split = placeholder_split
        print("Warning: Failed to import split from .formula, using placeholder.")

    rep, formulas = split(sentence)
    
    v_text = _text_model.encode(rep, normalize_embeddings=True)
    
    if formulas:
        # Ensure formulas is a list of strings
        formulas_str = [str(f) for f in formulas]
        v_math_embeddings = _math_model.encode(formulas_str, normalize_embeddings=True)
        if v_math_embeddings.ndim == 1: # handles case of single formula
             v_math = v_math_embeddings
        else:
             v_math = v_math_embeddings.mean(axis=0)
    else:
        v_math = np.zeros(MATH_DIM, dtype='float32') # Ensure correct dtype and shape
        
    # Ensure both vectors are 1D and then concatenate
    if v_text.ndim > 1:
        v_text = v_text.squeeze()
    if v_math.ndim > 1:
        v_math = v_math.squeeze()


    vec = np.concatenate([v_text, v_math]).astype('float32')
    assert vec.shape[0] == CFG.embed_dim, \
           f"encode dim {vec.shape[0]} ≠ {CFG.embed_dim}  sample:{sentence[:50]}"
    return vec
