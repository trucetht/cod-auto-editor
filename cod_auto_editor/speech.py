import re
from typing import List, Tuple, Optional
from .intervals import merge_intervals, clamp

_FALLBACK_FILLERS = ["um", "uh", "umm", "uhh", "uhm", "erm", "mm", "hmm"]
_FILLER_REGEXES = [
    re.compile(r"^u+h+$"),
    re.compile(r"^um+m+$"),
    re.compile(r"^er+m+$"),
    re.compile(r"^m+h*m+$"),
]

def _is_filler_token(token: str, fillers_set: set) -> bool:
    t = token.lower().strip().strip(".,!?;:\"'()[]{}")
    if t in fillers_set:
        return True
    if (t.startswith("uh") or t.startswith("um")) and 2 <= len(t) <= 6:
        return True
    for rgx in _FILTER_REGEXES:
        if rgx.match(t):
            return True
    return False

# fix variable name typo (_FILLER_REGEXES)
_FILTER_REGEXES = _FILLER_REGEXES

def detect_filler_segments(segments: List[dict], cfg: dict, video_duration: float) -> List[Tuple[float,float]]:
    fillers = set([f.lower().strip() for f in (cfg.get("filler_words") or []) if f.strip()]) or set(_FALLBACK_FILLERS)
    pre_pad = float(cfg.get("filler_pre_pad_sec", 0.25))
    post_pad = float(cfg.get("filler_post_pad_sec", 0.20))
    merge_gap = float(cfg.get("filler_merge_gap_sec", 0.30))

    raw = []
    for seg in segments:
        for w in seg.get("words", []):
            txt = (w.get("word") or "").lower().strip()
            txt_norm = txt.strip(".,!?;:\"'()[]{}")
            if _is_filler_token(txt_norm, fillers):
                s = clamp(float(w["start"]) - pre_pad, 0.0, video_duration)
                e = clamp(float(w["end"]) + post_pad, 0.0, video_duration)
                if e > s:
                    raw.append((s,e))
    return merge_intervals(raw, merge_gap)

def build_utterances_from_words(segments: List[dict],
                                max_gap_sec: float = 0.6) -> List[dict]:
    utterances = []
    cur_words = []
    cur_start = None
    last_end = None

    for seg in segments:
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w or not w.get("word"):
                continue
            ws, we = float(w["start"]), float(w["end"])
            if cur_start is None:
                cur_start = ws
            if last_end is not None and ws - last_end > max_gap_sec:
                if cur_words:
                    text = " ".join([cw["word"] for cw in cur_words]).strip()
                    utterances.append({"start": cur_start, "end": last_end, "text": text, "words": cur_words[:]})
                cur_words = []
                cur_start = ws
            cur_words.append({"word": w["word"], "start": ws, "end": we})
            last_end = we

    if cur_words:
        text = " ".join([cw["word"] for cw in cur_words]).strip()
        utterances.append({"start": cur_start, "end": last_end, "text": text, "words": cur_words[:]})

    if not utterances and segments:
        for seg in segments:
            s = float(seg["start"]); e = float(seg["end"])
            if e > s:
                utterances.append({"start": s, "end": e, "text": seg.get("text","").strip(), "words":[]})
    return utterances

def merge_tiny_utterances(utterances: List[dict], min_words: int = 4, join_gap_sec: float = 0.8) -> List[dict]:
    out = []
    buf = None
    for u in utterances:
        wc = len(u.get("words", []))
        if buf is None:
            buf = u
            continue
        if (wc < min_words or len(buf.get("words", [])) < min_words) and (u["start"] - buf["end"]) <= join_gap_sec:
            buf = {
                "start": buf["start"],
                "end": u["end"],
                "words": (buf.get("words", []) + u.get("words", [])),
                "text": (buf.get("text","") + " " + u.get("text","")).strip(),
            }
        else:
            out.append(buf)
            buf = u
    if buf is not None:
        out.append(buf)
    return out

def detect_silence_cuts_from_words(segments: List[dict],
                                   duration: float,
                                   min_gap_sec: float = 3.0,
                                   pre_pad: float = 0.05,
                                   post_pad: float = 0.05):
    word_times = []
    for seg in segments:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                word_times.append((float(w["start"]), float(w["end"])))
    word_times.sort(key=lambda x: x[0])

    cuts = []
    if not word_times:
        if duration >= min_gap_sec:
            cuts.append((0.0, duration))
        return cuts

    first_start = word_times[0][0]
    last_end = word_times[-1][1]

    if first_start >= min_gap_sec:
        cs = 0.0
        ce = max(0.0, first_start - post_pad)
        if ce > cs:
            cuts.append((cs, ce))

    for i in range(len(word_times) - 1):
        gap_start = word_times[i][1]
        gap_end = word_times[i+1][0]
        if (gap_end - gap_start) >= min_gap_sec:
            cs = max(0.0, gap_start + pre_pad)
            ce = max(0.0, gap_end - post_pad)
            if ce > cs:
                cuts.append((cs, ce))

    if (duration - last_end) >= min_gap_sec:
        cs = max(0.0, last_end + pre_pad)
        ce = duration
        if ce > cs:
            cuts.append((cs, ce))

    return merge_intervals(cuts, min_gap=0.05)
