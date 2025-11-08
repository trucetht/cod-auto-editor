# cod_auto_editor/analysis.py
from typing import List, Tuple, Optional, Callable
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from .intervals import merge_intervals, clamp

def compute_motion_array(
    video_path: str,
    fps_sample: int = 5,
    cb_frame: Optional[Callable[[np.ndarray, float], None]] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (times_sec, motion_norm, sample_fps)
    motion_norm ~ [0..1] based on frame-to-frame grayscale absdiff.

    If cb_frame is provided, it's called periodically as:
        cb_frame(frame_bgr, t_sec)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / max(1, fps_sample))))
    idx = 0

    prev = None
    vals: List[float] = []
    times: List[float] = []

    # Send a preview roughly every N grabbed frames (kept light)
    preview_stride = max(1, step)

    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))
            if prev is None:
                diff_val = 0.0
            else:
                diff = cv2.absdiff(gray, prev)
                diff_val = float(np.mean(diff)) / 255.0
            prev = gray
            t = (idx / src_fps) if src_fps else 0.0
            times.append(t)
            vals.append(diff_val)

            # live preview callback
            if cb_frame and (idx % preview_stride == 0):
                try:
                    cb_frame(frame, t)
                except Exception:
                    pass
        idx += 1

    cap.release()
    if not vals:
        return np.array([]), np.array([]), float(fps_sample)

    x = np.asarray(vals, dtype=np.float32)
    # normalize robustly
    p95 = float(np.quantile(x, 0.95)) if len(x) > 0 else 1.0
    scale = p95 if p95 > 1e-6 else (float(np.max(x)) or 1.0)
    x_norm = (x / scale).clip(0.0, 1.0)
    return np.asarray(times, dtype=np.float32), x_norm, (src_fps / step)

def compute_audio_rms_array(clip: VideoFileClip, step: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (times_sec, audio_rms_norm) sampled every `step` seconds.
    """
    if clip.audio is None:
        return np.array([]), np.array([])
    dur = float(clip.duration or 0.0)
    if dur <= 0:
        return np.array([]), np.array([])
    ts = np.arange(0.0, dur, step, dtype=np.float32)
    vals: List[float] = []
    for t in ts:
        a = clip.audio.to_soundarray(t, nbytes=2, fps=24000)
        if a is None or len(a) == 0:
            vals.append(0.0)
        else:
            arr = a.astype(np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            vals.append(float(np.sqrt(np.mean(arr * arr) + 1e-12)))
    x = np.asarray(vals, dtype=np.float32)
    p95 = float(np.quantile(x, 0.95)) if len(x) > 0 else 1.0
    scale = p95 if p95 > 1e-9 else (float(np.max(x)) or 1.0)
    x_norm = (x / scale).clip(0.0, 1.0)
    return ts, x_norm

def detect_downtime(t_audio: np.ndarray,
                    audio_norm: np.ndarray,
                    t_motion: np.ndarray,
                    motion_norm: np.ndarray,
                    cfg: dict) -> List[Tuple[float, float]]:
    """
    Mark windows where both audio and motion are 'low'.
    Thresholds can be numeric in config or 'auto'.
    """
    if t_audio.size == 0 or t_motion.size == 0:
        return []

    tm = t_motion
    mm = motion_norm
    ta = t_audio
    am = audio_norm

    interp_motion = np.interp(ta, tm, mm, left=mm[0], right=mm[-1])

    a_thr = cfg.get("audio_threshold", "auto")
    m_thr = cfg.get("motion_threshold", "auto")

    if isinstance(a_thr, (int, float)):
        thr_a = float(a_thr)
    else:
        thr_a = float(np.quantile(am, 0.25))

    if isinstance(m_thr, (int, float)):
        thr_m = float(m_thr)
    else:
        thr_m = float(np.quantile(interp_motion, 0.25))

    low = (am <= thr_a) & (interp_motion <= thr_m)

    step = float(ta[1] - ta[0]) if len(ta) > 1 else 0.5
    min_len = float(cfg.get("min_downtime_sec", 3.0))
    min_frames = max(1, int(round(min_len / max(1e-6, step))))

    cuts: List[Tuple[float, float]] = []
    run_start = None
    for i, flag in enumerate(low):
        if flag and run_start is None:
            run_start = ta[i]
        elif (not flag) and run_start is not None:
            run_end = ta[i]
            if i - int(np.searchsorted(ta, run_start)) >= min_frames:
                cuts.append((float(run_start), float(run_end)))
            run_start = None
    if run_start is not None:
        run_end = ta[-1] + step
        if len(ta) - int(np.searchsorted(ta, run_start)) >= min_frames:
            cuts.append((float(run_start), float(run_end)))

    return merge_intervals(cuts, min_gap=0.10)
