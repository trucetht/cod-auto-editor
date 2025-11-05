# cod_auto_editor/intro.py
from typing import List, Tuple, Optional
from .intervals import tok, normalize_text, text_sim_fallback, merge_intervals, clamp, _PUNC_STRIP_TBL, _fmt_time
from .utils import details_path

def _flatten_words_from_segments(segments: List[dict], window_sec: float) -> List[dict]:
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            if "start" not in w or "word" not in w or w["word"] is None:
                continue
            t = float(w["start"])
            if t > window_sec:
                return words
            words.append({"word": w["word"], "start": t, "end": float(w.get("end", t))})
    return words

def _find_phrase_hits_on_words(words: List[dict], phrases: List[str], token_gap_sec: float) -> List[Tuple[float,int]]:
    if not words:
        return []
    phrases_tok = [(tok(p), p) for p in phrases if p.strip()]
    hits: List[Tuple[float,int]] = []
    for pvec, _ in phrases_tok:
        if not pvec:
            continue
        m = len(pvec)
        j = 0
        last_t = None
        start_idx = None
        for i, w in enumerate(words):
            token = (w["word"] or "").translate(_PUNC_STRIP_TBL).lower()
            if token == pvec[j]:
                if j == 0:
                    start_idx = i
                    last_t = w["start"]
                    j = 1 if m > 1 else m
                else:
                    if w["start"] - (last_t or w["start"]) <= token_gap_sec:
                        last_t = w["start"]
                        j += 1
                    else:
                        start_idx = i
                        last_t = w["start"]
                        j = 1 if m > 1 else m
                if j >= m:
                    hits.append((words[start_idx]["start"], start_idx))
                    j = 0
                    last_t = None
                    start_idx = None
            else:
                if j > 0 and token == pvec[0]:
                    start_idx = i
                    last_t = w["start"]
                    j = 1 if m > 1 else m
                    if j >= m:
                        hits.append((words[start_idx]["start"], start_idx))
                        j = 0
                        last_t = None
                        start_idx = None
                else:
                    j = 0
                    last_t = None
                    start_idx = None
    hits.sort(key=lambda x: x[0])
    deduped = []
    for t, idx in hits:
        if not deduped or (t - deduped[-1][0]) > 0.75:
            deduped.append((t, idx))
    return deduped

def _enclosing_utterance_start(t: float, utterances: List[dict]) -> Optional[float]:
    for u in utterances:
        if float(u["start"]) - 0.001 <= t <= float(u["end"]) + 0.001:
            return float(u["start"])
    future = [float(u["start"]) for u in utterances if float(u["start"]) >= t - 0.001]
    return min(future) if future else None

def find_intro_anchor_cut_and_retakes(
    utterances: List[dict], cfg: dict, duration: float, segments_for_words: Optional[List[dict]] = None
) -> Tuple[Optional[Tuple[float,float]], List[Tuple[float,float]]]:
    window      = float(cfg.get("intro_anchor_window_sec", 70.0))
    min_repeat  = int(cfg.get("intro_anchor_min_repeat", 2))
    sim_thresh  = float(cfg.get("intro_anchor_similarity", 0.84))
    token_gap   = float(cfg.get("intro_anchor_token_gap_sec", 3.0))
    phrases     = cfg.get("intro_anchor_phrases") or ["welcome back", "welcome back guys", "all right welcome back"]
    snap_mode   = (cfg.get("intro_anchor_snap_to") or "word").lower()  # "word" | "utterance"
    prepad      = float(cfg.get("intro_anchor_word_snap_prepad_sec", 0.0))

    words = _flatten_words_from_segments(segments_for_words or [], window)
    hits = _find_phrase_hits_on_words(words, phrases, token_gap_sec=token_gap)

    dbg = details_path(cfg, "intro_debug.txt")
    with open(dbg, "w", encoding="utf-8") as f:
        f.write(f"# Intro debug\nwindow={window}s, min_repeat={min_repeat}, phrases={phrases}\n")
        if hits:
            for t, _ in hits:
                f.write(f"HIT(word-stream) @ {t:.2f}s\n")
        else:
            f.write("No word-stream hits.\n")

    if len(hits) >= min_repeat:
        per_cuts = []
        for t, _ in hits[:-1]:
            us = _enclosing_utterance_start(t, utterances)
            ue = None
            for u in utterances:
                if float(u["start"]) == us:
                    ue = float(u["end"]); break
            if us is not None and ue is not None and ue > us:
                per_cuts.append((us, ue))

        last_t = hits[-1][0]
        if snap_mode == "word":
            head_end = clamp(max(0.0, last_t - prepad), 0.0, duration)
        else:
            snap_start = _enclosing_utterance_start(last_t, utterances)
            head_end = clamp(snap_start if snap_start is not None else last_t, 0.0, duration)

        head_cut = (0.0, head_end)

        with open(dbg, "a", encoding="utf-8") as f:
            f.write(f"Resolved head-cut 0→{head_cut[1]:.2f}s (mode={snap_mode}).\n")
            f.write(f"Per-intro removals: {len(per_cuts)}\n")
        return head_cut, per_cuts

    # Semantic fallback
    cand = [u for u in utterances if float(u["start"]) <= window and len(normalize_text(u.get("text",""))) > 0]
    if len(cand) < min_repeat:
        with open(dbg, "a", encoding="utf-8") as f:
            f.write(f"Not enough early utterances for semantic fallback: {len(cand)} < {min_repeat}\n")
        return None, []

    texts = [normalize_text(u["text"]) for u in cand]
    n = len(texts)
    similar_sets: List[List[int]] = [[] for _ in range(n)]

    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        for i in range(n):
            for j in range(i+1, n):
                sim = float(util.cos_sim(embs[i], embs[j]))
                if sim >= sim_thresh:
                    similar_sets[i].append(j)
                    similar_sets[j].append(i)
    except Exception:
        for i in range(n):
            for j in range(i+1, n):
                sim = text_sim_fallback(texts[i], texts[j])
                if sim >= (sim_thresh - 0.04):
                    similar_sets[i].append(j)
                    similar_sets[j].append(i)

    groups = []
    used = set()
    for i, neigh in enumerate(similar_sets):
        if i in used: continue
        grp = sorted(set([i] + neigh))
        if len(grp) >= min_repeat:
            groups.append(grp); used.update(grp)

    if not groups:
        with open(dbg, "a", encoding="utf-8") as f:
            f.write("Semantic fallback found no groups.\n")
        return None, []

    def last_time_of_group(g):
        return float(cand[max(g)]["start"])
    best = max(groups, key=last_time_of_group)

    per_cuts = []
    for idx in best[:-1]:
        s = float(cand[idx]["start"]); e = float(cand[idx]["end"])
        if e > s:
            per_cuts.append((s,e))

    last_start = float(cand[best[-1]]["start"])
    head_cut = (0.0, clamp(last_start, 0.0, duration))
    with open(dbg, "a", encoding="utf-8") as f:
        f.write(f"Semantic fallback → head-cut 0→{head_cut[1]:.2f}s; per-intro cuts={len(per_cuts)}\n")
    return head_cut, per_cuts
