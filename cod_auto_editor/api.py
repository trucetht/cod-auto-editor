# cod_auto_editor/api.py
from __future__ import annotations
import os
from typing import Callable, Optional, Tuple, List

from .db import pg_connect_from_env, load_triggers
from .asr import transcribe_audio
from .speech import (
    detect_filler_segments,
    build_utterances_from_words,
    merge_tiny_utterances,
    detect_silence_cuts_from_words,
)
from .intro import find_intro_anchor_cut_and_retakes
from .dedupe import detect_semantic_duplicates, log_semantic_dupes
from .analysis import compute_motion_array, compute_audio_rms_array, detect_downtime
from .hitmarker import detect_hitmarker_events, build_chained_keep_windows_from_events
from .edit import build_edit_pieces, build_keep_only_pieces
from .renderers import render_with_moviepy, render_with_ffmpeg_hdr
from .utils import save_transcript_txt, save_intervals_txt
from moviepy.editor import VideoFileClip

def _p(cb: Optional[Callable[[float], None]], x: float):
    try:
        if cb: cb(max(0.0, min(1.0, float(x))))
    except Exception:
        pass

def _l(cb: Optional[Callable[[str], None]], msg: str):
    try:
        if cb: cb(msg)
        else: print(msg)
    except Exception:
        pass

def run_pipeline(
    input_path: str,
    config_path: str,
    *,
    log: Optional[Callable[[str], None]] = None,
    progress: Optional[Callable[[float], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> dict:
    import yaml
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    def cancelled():
        try:
            return bool(is_cancelled and is_cancelled())
        except Exception:
            return False

    os.makedirs("output", exist_ok=True)
    _l(log, "[Init] Starting pipeline…")
    _p(progress, 0.02)

    # Optional overlays
    triggers, assets = [], {}
    if cfg.get("enable_overlays", False) and not cfg.get("hdr_preserve", False):
        try:
            _l(log, "[DB] Loading overlay triggers/assets…")
            conn = pg_connect_from_env()
            triggers, assets = load_triggers(conn)
        except Exception as e:
            _l(log, f"[DB] Skipping overlays DB init: {e}")

    if cancelled(): raise RuntimeError("Cancelled")

    # Hitmarker keep windows (optional)
    keep_from_hits: List[Tuple[float, float]] = []
    events = []
    if bool(cfg.get("enable_hitmarker_filter", False)):
        _l(log, "[Hitmarker] Detecting hit/gun events…")
        events, spf = detect_hitmarker_events(input_path, cfg)
        if events:
            pre_buf = float(cfg.get("hit_pre_buffer_sec", 0.30))
            post_buf = float(cfg.get("hit_post_buffer_sec", 5.00))
            chain_gap = float(cfg.get("hit_chain_gap_sec", 2.0))
            with VideoFileClip(input_path) as probe:
                dur_tmp = probe.duration or 0.0
            keep_from_hits = build_chained_keep_windows_from_events(
                events, pre_buf, post_buf, chain_gap, dur_tmp
            )
            _l(log, f"[Hitmarker] Events={len(events)} → keep windows={len(keep_from_hits)}")
        else:
            _l(log, "[Hitmarker] No events found with current thresholds.")
    _p(progress, 0.10)
    if cancelled(): raise RuntimeError("Cancelled")

    # Transcribe original
    _l(log, "[ASR] Transcribing original…")
    segments_orig = transcribe_audio(input_path, cfg, purpose="original")
    save_transcript_txt("output/details/transcript_original.txt", segments_orig, "Original transcript")
    _p(progress, 0.18)
    if cancelled(): raise RuntimeError("Cancelled")

    # Duration
    with VideoFileClip(input_path) as clip:
        duration = clip.duration or 0.0

    # Filler
    filler_cuts = detect_filler_segments(segments_orig, cfg, duration)
    save_intervals_txt("output/details/filler_cuts.txt", filler_cuts, "Filler word cut windows")
    _l(log, f"[Filler] windows={len(filler_cuts)}")

    # Silence
    sil_min  = float(cfg.get("silence_cut_min_gap_sec", cfg.get("min_silence_cut_sec", 3.0)))
    sil_pre  = float(cfg.get("silence_pre_pad_sec", 0.05))
    sil_post = float(cfg.get("silence_post_pad_sec", 0.05))
    silence_cuts = detect_silence_cuts_from_words(segments_orig, duration, sil_min, sil_pre, sil_post)
    save_intervals_txt("output/details/silence_cuts.txt", silence_cuts, "Silence (no-speech) cut windows")
    _l(log, f"[Silence] (≥{sil_min}s) windows={len(silence_cuts)}")
    _p(progress, 0.28)
    if cancelled(): raise RuntimeError("Cancelled")

    # Utterances
    utt_gap = float(cfg.get("utterance_max_gap_sec", 1.2))
    utterances = build_utterances_from_words(segments_orig, max_gap_sec=utt_gap)
    utterances = merge_tiny_utterances(utterances, min_words=4, join_gap_sec=0.8)

    # Intro anchor
    head_cut, per_cuts = find_intro_anchor_cut_and_retakes(utterances, cfg, duration, segments_for_words=segments_orig)
    intro_cuts = []
    if head_cut:
        intro_cuts.append(head_cut)
        save_intervals_txt("output/details/intro_anchor.txt", intro_cuts, "Intro anchor cut(s)")
    _p(progress, 0.34)
    if cancelled(): raise RuntimeError("Cancelled")

    # Semantic dedupe
    sem_enable   = bool(cfg.get("enable_semantic_dedupe", True))
    sem_thresh   = float(cfg.get("dedupe_similarity", 0.88))
    sem_window   = float(cfg.get("dedupe_window_sec", 180.0))
    sem_pre_pad  = float(cfg.get("dedupe_pre_pad_sec", 0.15))
    sem_post_pad = float(cfg.get("dedupe_post_pad_sec", 0.15))

    semantic_dupe_cuts = []
    if sem_enable and utterances:
        _l(log, "[Dedupe] Detecting semantic duplicates…")
        semantic_dupe_cuts = detect_semantic_duplicates(
            utterances, duration, sem_thresh, sem_window, sem_pre_pad, sem_post_pad
        )
    save_intervals_txt("output/details/semantic_dupes.txt", semantic_dupe_cuts, "Semantic duplicate cut windows")
    log_semantic_dupes("output/details/semantic_dupes_detail.txt", utterances, semantic_dupe_cuts)
    _p(progress, 0.42)
    if cancelled(): raise RuntimeError("Cancelled")

    # Motion/audio downtime (best-effort)
    downtime_short = []; downtime_long = []
    try:
        _l(log, "[Analysis] Computing motion/audio metrics…")
        t_motion, motion_norm, _ = compute_motion_array(input_path, fps_sample=int(cfg.get("analysis_fps", 5)))
        with VideoFileClip(input_path) as clip2:
            t_audio, audio_norm = compute_audio_rms_array(clip2, step=0.5)
        downtime_all = detect_downtime(t_audio, audio_norm, t_motion, motion_norm, cfg)
        short_thresh = float(cfg.get("downtime_short_threshold_sec", 10.0))
        downtime_short = [(s,e) for (s,e) in downtime_all if (e - s) <= short_thresh]
        downtime_long  = [(s,e) for (s,e) in downtime_all if (e - s)  > short_thresh]
        save_intervals_txt("output/details/downtime_short.txt", downtime_short, "Downtime short (speed ramp)")
        save_intervals_txt("output/details/downtime_long.txt",  downtime_long,  "Downtime long (jump cut)")
        _l(log, f"[Downtime] short={len(downtime_short)} long={len(downtime_long)}")
    except Exception as e:
        _l(log, f"[Analysis] Skipping downtime: {e}")
    _p(progress, 0.55)
    if cancelled(): raise RuntimeError("Cancelled")

    # Build sections
    if keep_from_hits:
        pieces = build_keep_only_pieces(duration, keep_from_hits)
    else:
        from .intervals import merge_intervals
        hard_first_pass = merge_intervals(
            (intro_cuts if intro_cuts else []) + filler_cuts + semantic_dupe_cuts, min_gap=0.05
        )
        pieces = build_edit_pieces(duration, hard_first_pass, silence_cuts, downtime_short, downtime_long)

    # Render
    if cfg.get("hdr_preserve", False):
        _l(log, "[Render] HDR-preserve via FFmpeg (10-bit)…")
        out_path = render_with_ffmpeg_hdr(input_path, cfg, pieces)
    else:
        overlays = []
        if cfg.get("enable_overlays", False) and triggers and assets:
            from .overlays import find_overlay_events
            _l(log, "[Overlays] Matching triggers…")
            overlays = find_overlay_events(segments_orig, triggers, assets, cfg)
        _l(log, "[Render] MoviePy SDR path…")
        out_path = render_with_moviepy(input_path, cfg, pieces, overlays)
    _p(progress, 0.86)
    if cancelled(): raise RuntimeError("Cancelled")

    # Transcribe final for verification
    _l(log, "[ASR] Transcribing final output…")
    segments_final = transcribe_audio(out_path, cfg, purpose="final")
    save_transcript_txt("output/details/transcript_final.txt", segments_final, "Final transcript")

    _p(progress, 1.0)
    _l(log, f"[Done] {out_path}")
    return {
        "output_path": out_path,
        "pieces_count": len(pieces),
        "events_count": len(events),
    }
