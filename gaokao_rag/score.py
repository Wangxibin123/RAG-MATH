import numpy as np
from .cfg import CFG

def hybrid(rerank_score: float, difficulty: float = None) -> float:
    """
    Calculates a hybrid score combining rerank_score and difficulty.

    Args:
        rerank_score (float): The semantic similarity score, already in [0, 1].
        difficulty (float, optional): The difficulty of the item, in [0, 100]. 
                                       If None, difficulty penalty is not applied.

    Returns:
        float: The final hybrid score.
    """
    d0 = CFG.base['difficulty']['default']
    coeff = CFG.diff_coeff # direct access as per new Cfg structure

    if difficulty is None or coeff == 0: # If no difficulty or coeff is 0, only rerank_score matters
        return rerank_score

    # Normalize difficulty penalty to be in [0, 1]
    # Penalty is 0 if difficulty == d0, and 1 if difficulty is 0 or 100 (assuming d0 is between 0 and 100)
    # max_diff = max(d0, 100 - d0)
    # diff_penalty = abs(difficulty - d0) / max_diff if max_diff !=0 else 0 
    # Simpler penalty as per original: abs(difficulty - d0) / 100
    diff_penalty = abs(difficulty - d0) / 100.0
    
    # Original formula: (1-coeff)*rerank_score - coeff*diff_penalty
    # This means higher penalty REDUCES the score.
    # If rerank_score is high (e.g., 1) and penalty is high (e.g., 1), and coeff is high (e.g., 0.7)
    # Score = (0.3*1) - (0.7*1) = -0.4. Scores can be negative.
    final_score = (1 - coeff) * rerank_score - coeff * diff_penalty
    
    return final_score 