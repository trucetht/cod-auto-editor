from typing import List, Tuple
from .intervals import normalize_text, text_sim_fallback, merge_intervals, clamp

def detect_semantic_duplicates(utterances: List[dict],
                               video_duration: float,
                               sim_thresh: float = 0.88,
                               lookback_sec: float = 180.0,
                               pre_pad: float = 0.15,
                               post_pad: float = 0.15) -> List[Tuple[float, float]]:
    if not utterances:
        print("[Dedupe] No utterances; skipping.")
        return []

    # Try sentence-transformers; fallback to text_sim_fallback
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [normalize_text(u.get("text","")) for u in utterances]
        lens = [len(t.split()) for t in texts]
        keep_idx = [i for i,l in enumerate(lens) if l > 3]
        if not keep_idx:
            print("[Dedupe] All utterances too short; skipping.")
            return []
        embs = model.encode([texts[i] for i in keep_idx], convert_to_tensor=True, normalize_embeddings=True)

        cuts = []
        matches = 0
        for ii, i in enumerate(keep_idx):
            ui = utterances[i]; si, ei = float(ui["start"]), float(ui["end"])
            for jj in range(ii-1, -1, -1):
                j = keep_idx[jj]
                uj = utterances[j]; sj, ej = float(uj["start"]), float(uj["end"])
                if si - ej > lookback_sec: break
                if (ej - sj) < 0.6: continue
                sim = float(util.cos_sim(embs[ii], embs[jj]))
                if sim >= sim_thresh:
                    matches += 1
                    cut_s = clamp(sj - pre_pad, 0.0, video_duration)
                    cut_e = clamp(ej + post_pad, 0.0, video_duration)
                    if cut_e > cut_s:
                        cuts.append((cut_s, cut_e))
        cuts = merge_intervals(cuts, min_gap=0.10)
        print(f"[Dedupe] Found {matches} near-duplicate pairs; cutting {len(cuts)} windows.")
        return cuts
    except Exception:
        cuts = []
        matches = 0
        for i in range(len(utterances)):
            ui = utterances[i]; si, ei = float(ui["start"]), float(ui["end"])
            for j in range(i-1, -1, -1):
                uj = utterances[j]; sj, ej = float(uj["start"]), float(uj["end"])
                if si - ej > lookback_sec:
                    break
                if (ej - sj) < 0.6:
                    continue
                sim = text_sim_fallback(ui.get("text",""), uj.get("text",""))
                if sim >= (sim_thresh - 0.06):
                    matches += 1
                    cuts.append((clamp(sj - pre_pad, 0.0, video_duration),
                                 clamp(ej + post_pad, 0.0, video_duration)))
        cuts = merge_intervals(cuts, min_gap=0.10)
        print(f"[Dedupe] (fallback) Found {matches} near-duplicate pairs; cutting {len(cuts)} windows.")
        return cuts

def log_semantic_dupes(path: str, utterances: List[dict],
                       dup_cuts: List[Tuple[float,float]], fmt_time=None):
    import os
    os.makedirs("output", exist_ok=True)
    from .utils import _fmt_time as _fmt
    _fmt_time = fmt_time or _fmt
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Semantic duplicate removals (keep later takes)\n")
        for (cs, ce) in dup_cuts:
            f.write(f"\n## CUT {_fmt_time(cs)} -> {_fmt_time(ce)} ({ce-cs:.2f}s)\n")
            for u in utterances:
                us, ue = float(u["start"]), float(u["end"])
                if us >= cs and ue <= ce:
                    f.write(f"- [{_fmt_time(us)} → {_fmt_time(ue)}] {u.get('text','')[:200]}\n")
