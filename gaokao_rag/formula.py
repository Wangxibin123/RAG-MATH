import regex as re

# Pattern to find LaTeX blocks: $$...$$, \[...\] (display math), and $...$ (inline math)
# It handles nested structures correctly due to regex module's capabilities.
# re.S (DOTALL) allows . to match newlines, important for multi-line formulas.
PATTERN = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]|\$.*?\$", re.S)

def split(text: str):
    formulas = []
    def repl(match_obj):
        formulas.append(match_obj.group(0))
        return f"[M{len(formulas)-1}]" # Placeholder like [M0], [M1], ...
    
    processed_text = PATTERN.sub(repl, text)
    return processed_text, formulas