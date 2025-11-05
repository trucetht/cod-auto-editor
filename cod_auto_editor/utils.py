# cod_auto_editor/utils.py
import os
from typing import List, Tuple

# ---------- Output/Details directory helpers ----------

def get_output_dirs(cfg: dict | None = None) -> tuple[str, str]:
    """
    Returns (output_dir, details_dir). Creates them if missing.
    Defaults: output_dir="output", details_dir="output/details".
    You can override via config:
      output_dir: "<path>"
      details_dir_name: "details"
    """
    base = (cfg or {}).get("output_dir", "output")
    details_name = (cfg or {}).get("details_dir_name", "details")
    base_abs = os.path.abspath(base)
    details_abs = os.path.join(base_abs, details_name)
    os.makedirs(base_abs, exist_ok=True)
    os.makedirs(details_abs, exist_ok=True)
    return base_abs, details_abs

def details_path(cfg: dict | None, filename: str) -> str:
    """
    Returns full path under output/details/<filename> and ensures dirs exist.
    """
    _out, det = get_output_dirs(cfg)
    p = os.path.join(det, filename)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p

def out_video_dir(cfg: dict | None) -> str:
    """
    Returns the output (top-level) directory where MP4 files are placed.
    """
    out, _det = get_output_dirs(cfg)
    return out

# ---------- Simple save helpers (write into any given path) ----------

def save_transcript_txt(path: str, segments: List[dict], title: str = "Transcript") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        for seg in segments:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", s))
            text = (seg.get("text") or "").strip()
            f.write(f"[{_fmt_time(s)} → {_fmt_time(e)}] {text}\n")

def save_intervals_txt(path: str, intervals: List[Tuple[float,float]], title: str = "Intervals") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        for s, e in intervals:
            f.write(f"{_fmt_time(s)} -> {_fmt_time(e)} ({e - s:.2f}s)\n")

# local utility (duplicated here so helpers can work standalone)
def _fmt_time(t: float) -> str:
    total = float(t)
    hh = int(total // 3600)
    mm = int((total % 3600) // 60)
    ss = total % 60
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:06.3f}"
    return f"{mm:02d}:{ss:06.3f}"
