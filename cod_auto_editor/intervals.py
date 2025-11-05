# cod_auto_editor/intervals.py
from typing import List, Tuple
import string

# ---------- Interval math ----------

def merge_intervals(intervals: List[Tuple[float, float]], min_gap: float) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    ints = sorted([(min(a, b), max(a, b)) for a, b in intervals])
    merged = []
    cs, ce = ints[0]
    for s, e in ints[1:]:
        if s <= ce + min_gap:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged

def subtract_intervals(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out = []
    for s, e in a:
        cur = [(s, e)]
        for bs, be in b:
            nxt = []
            for xs, xe in cur:
                if be <= xs or bs >= xe:
                    nxt.append((xs, xe))
                else:
                    if xs < bs:
                        nxt.append((xs, bs))
                    if be < xe:
                        nxt.append((be, xe))
            cur = nxt
        out.extend(cur)
    return [(x, y) for x, y in out if y - x > 1e-6]

def clamp(a: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, a))

# ---------- Text helpers (used by intro/dedupe) ----------

_PUNC_STRIP_TBL = str.maketrans("", "", string.punctuation)

def tok(s: str) -> List[str]:
    if not s:
        return []
    # lower, strip punctuation, split on whitespace
    s2 = s.lower().translate(_PUNC_STRIP_TBL)
    return [t for t in s2.split() if t]

def normalize_text(s: str) -> str:
    return " ".join(tok(s))

def text_sim_fallback(a: str, b: str) -> float:
    """
    Simple token Jaccard similarity as a robust fallback.
    """
    ta = set(tok(a))
    tb = set(tok(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / max(1, union)

# ---------- Small time formatter used in logs/vtt ----------

def _fmt_time(t: float) -> str:
    total = float(t)
    hh = int(total // 3600)
    mm = int((total % 3600) // 60)
    ss = total % 60
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:06.3f}"
    return f"{mm:02d}:{ss:06.3f}"
