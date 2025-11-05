# cod_auto_editor/cli.py
import argparse, os
from moviepy.editor import VideoFileClip
import yaml

from .db import pg_connect_from_env, load_triggers
from .asr import transcribe_audio
from .speech import detect_filler_segments, build_utterances_from_words, merge_tiny_utterances, detect_silence_cuts_from_words
from .intro import find_intro_anchor_cut_and_retakes
from .dedupe import detect_semantic_duplicates, log_semantic_dupes
from .analysis import compute_motion_array, compute_audio_rms_array, detect_downtime
from .hitmarker import (
    detect_hitmarker_events,
    build_chained_keep_windows_from_events,
    build_chained_keep_windows_from_labeled_events,
)
from .edit import build_edit_pieces, build_keep_only_pieces
from .renderers import render_with_moviepy, render_with_ffmpeg_hdr
from .overlays import find_overlay_events
from .intervals import merge_intervals
from .utils import save_transcript_txt, save_intervals_txt, details_path, out_video_dir
from .visual import classify_hitmarkers  # <-- NEW

def main():
    parser = argparse.ArgumentParser(description="COD Auto Editor (modular)")
    parser.add_argument("--input", required=False, default="input/match1.mp4", help="Path to input video")
    parser.add_argument("--config", default=os.environ.get("CONFIG_PATH", "config.yml"), help="Path to config.yml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = out_video_dir(cfg)
    _details_dir = details_path(cfg, ".keep")
    try:
        if os.path.exists(_details_dir):
            os.remove(_details_dir)
    except Exception:
        pass

    triggers = []
    assets = {}
    if cfg.get("enable_overlays", False) and not cfg.get("hdr_preserve", False):
        try:
            conn = pg_connect_from_env()
            triggers, assets = load_triggers(conn)
        except Exception as e:
            print("[DB] Skipping overlays DB init:", e)

    keep_from_hits = []
    events = []
    labels = []

    if bool(cfg.get("enable_hitmarker_filter", False)):
        print("Detecting hitmarker/gun events (audio)…")
        events, spf = detect_hitmarker_events(args.input, cfg)
        if events:
            print(f"[Hitmarker] Found {len(events)} audio events. Classifying visual color…")
            labels = classify_hitmarkers(args.input, events, cfg)

            # Write labeled CSV
            lbl_csv = details_path(cfg, f"{cfg.get('hit_probe_prefix','bo7_probe')}_events_with_labels.csv")
            import csv as _csv
            with open(lbl_csv, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["start_sec","end_sec","peak_sec","peak_high_band","label","red_frames","white_frames","red_ratio_max","white_ratio_max"])
                for (s,e,pt,pv), L in zip(events, labels):
                    w.writerow([
                        f"{s:.6f}", f"{e:.6f}", f"{pt:.6f}", f"{pv:.6f}",
                        L.get("label","unknown"),
                        L.get("red_frames",0),
                        L.get("white_frames",0),
                        f"{L.get('red_ratio_max',0.0):.6f}",
                        f"{L.get('white_ratio_max',0.0):.6f}",
                    ])

            # Build keep windows with per-label buffers
            kill_pre  = float(cfg.get("kill_pre_buffer_sec", 0.75))
            kill_post = float(cfg.get("kill_post_buffer_sec", 2.50))
            non_pre   = float(cfg.get("nonkill_pre_buffer_sec", 0.30))
            non_post  = float(cfg.get("nonkill_post_buffer_sec", 0.75))
            chain_gap = float(cfg.get("hit_chain_gap_sec", 3.5))

            # probe duration
            probe_clip_tmp = VideoFileClip(args.input)
            duration_tmp = probe_clip_tmp.duration or 0.0
            probe_clip_tmp.close()

            keep_from_hits = build_chained_keep_windows_from_labeled_events(
                events, labels, kill_pre, kill_post, non_pre, non_post, chain_gap, duration_tmp
            )
            print(f"[Hitmarker] Keeping {len(keep_from_hits)} chained window(s) based on color labels.")
        else:
            print("[Hitmarker] No events found with current thresholds.")

    print("Transcribing (ASR)…")
    segments_orig = transcribe_audio(args.input, cfg, purpose="original")
    save_transcript_txt(details_path(cfg, "transcript_original.txt"), segments_orig, "Original transcript")

    probe_clip = VideoFileClip(args.input)
    duration = probe_clip.duration or 0.0
    probe_clip.close()

    filler_cuts = detect_filler_segments(segments_orig, cfg, duration)
    print(f"[Filler] Cut windows: {len(filler_cuts)}")
    save_intervals_txt(details_path(cfg, "filler_cuts.txt"), filler_cuts, "Filler word cut windows")

    sil_min  = float(cfg.get("silence_cut_min_gap_sec", cfg.get("min_silence_cut_sec", 3.0)))
    sil_pre  = float(cfg.get("silence_pre_pad_sec", 0.05))
    sil_post = float(cfg.get("silence_post_pad_sec", 0.05))
    silence_cuts = detect_silence_cuts_from_words(
        segments_orig, duration, min_gap_sec=sil_min, pre_pad=sil_pre, post_pad=sil_post
    )
    print(f"[Silence] (≥{sil_min}s no speech) windows: {len(silence_cuts)}")
    save_intervals_txt(details_path(cfg, "silence_cuts.txt"), silence_cuts, "Silence (no-speech) cut windows")

    utt_gap = float(cfg.get("utterance_max_gap_sec", 1.2))
    utterances = build_utterances_from_words(segments_orig, max_gap_sec=utt_gap)
    utterances = merge_tiny_utterances(utterances, min_words=4, join_gap_sec=0.8)

    intro_head_cut, intro_per_cuts = find_intro_anchor_cut_and_retakes(
        utterances, cfg, duration, segments_for_words=segments_orig
    )
    intro_cuts = []
    if intro_head_cut:
        intro_cuts.append(intro_head_cut)
        save_intervals_txt(details_path(cfg, "intro_anchor.txt"), intro_cuts, "Intro anchor cut(s)")

    sem_enable   = bool(cfg.get("enable_semantic_dedupe", True))
    sem_thresh   = float(cfg.get("dedupe_similarity", 0.88))
    sem_window   = float(cfg.get("dedupe_window_sec", 180.0))
    sem_pre_pad  = float(cfg.get("dedupe_pre_pad_sec", 0.15))
    sem_post_pad = float(cfg.get("dedupe_post_pad_sec", 0.15))

    semantic_dupe_cuts = []
    if sem_enable and utterances:
        print("[Dedupe] Detecting semantic duplicates (keep latest take)…")
        from .dedupe import detect_semantic_duplicates, log_semantic_dupes
        semantic_dupe_cuts = detect_semantic_duplicates(
            utterances, video_duration=duration, sim_thresh=sem_thresh,
            lookback_sec=sem_window, pre_pad=sem_pre_pad, post_pad=sem_post_pad
        )
    save_intervals_txt(details_path(cfg, "semantic_dupes.txt"), semantic_dupe_cuts, "Semantic duplicate cut windows")
    log_semantic_dupes(details_path(cfg, "semantic_dupes_detail.txt"), utterances, semantic_dupe_cuts)

    downtime_short = []
    downtime_long  = []
    try:
        print("Analyzing motion/audio…")
        t_motion, motion_norm, _ = compute_motion_array(args.input, fps_sample=int(cfg.get("analysis_fps", 5)))
        probe_clip2 = VideoFileClip(args.input)
        t_audio, audio_norm = compute_audio_rms_array(probe_clip2, step=0.5)
        probe_clip2.close()
        downtime_all = detect_downtime(t_audio, audio_norm, t_motion, motion_norm, cfg)
        short_thresh = float(cfg.get("downtime_short_threshold_sec", 10.0))
        downtime_short = [(s,e) for (s,e) in downtime_all if (e - s) <= short_thresh]
        downtime_long  = [(s,e) for (s,e) in downtime_all if (e - s)  > short_thresh]
        print(f"[Downtime] short (≤{short_thresh}s): {len(downtime_short)}; long: {len(downtime_long)}")
        save_intervals_txt(details_path(cfg, "downtime_short.txt"), downtime_short, "Downtime short (speed ramp)")
        save_intervals_txt(details_path(cfg, "downtime_long.txt"),  downtime_long,  "Downtime long (jump cut)")
    except Exception as e:
        print("[Analysis] Skipping downtime analysis:", e)

    if keep_from_hits:
        pieces = build_keep_only_pieces(duration, keep_from_hits)
    else:
        hard_first_pass = merge_intervals((intro_cuts if intro_cuts else []) + filler_cuts + semantic_dupe_cuts, min_gap=0.05)
        pieces = build_edit_pieces(duration, hard_first_pass, silence_cuts, downtime_short, downtime_long)

    if cfg.get("hdr_preserve", False):
        print("Rendering (HDR-preserve via FFmpeg 10-bit)…")
        out = render_with_ffmpeg_hdr(args.input, cfg, pieces)
    else:
        overlays = []
        if cfg.get("enable_overlays", False):
            print("Matching overlay triggers…")
            overlays = find_overlay_events(segments_orig, triggers, assets, cfg)
        print("Rendering (MoviePy SDR path)…")
        out = render_with_moviepy(args.input, cfg, pieces, overlays)

    print(f"Done → {out}")

    print("Transcribing final output for verification (ASR)…")
    segments_final = transcribe_audio(out, cfg, purpose="final")
    save_transcript_txt(details_path(cfg, "transcript_final.txt"), segments_final, "Final transcript")

    print("Saved:")
    print("  output/details/transcript_original.txt")
    print("  output/details/filler_cuts.txt")
    print("  output/details/silence_cuts.txt")
    print("  output/details/intro_anchor.txt" if intro_cuts else "  (no intro anchor cut)")
    print("  output/details/semantic_dupes.txt")
    print("  output/details/semantic_dupes_detail.txt")
    if downtime_short:
        print("  output/details/downtime_short.txt")
    if downtime_long:
        print("  output/details/downtime_long.txt")
    print("  output/details/transcript_final.txt")
    if bool(cfg.get("enable_hitmarker_filter", False)) and events:
        pref = cfg.get("hit_probe_prefix", "bo7_probe")
        print(f"  output/details/{pref}_frames.csv")
        print(f"  output/details/{pref}_events.csv")
        print(f"  output/details/{pref}_events_with_labels.csv")
        print(f"  output/details/{pref}_transcript.txt")
        print(f"  output/details/{pref}_markers.vtt")

if __name__ == "__main__":
    main()
