# cod_auto_editor/hitmarker.py
import os, subprocess, numpy as np
from typing import List, Tuple
from .intervals import merge_intervals, clamp, _fmt_time as fmt_time
from .utils import details_path

def _ffmpeg_bin() -> str:
    return "ffmpeg"

def _export_mono_wav(input_path: str, out_wav: str, sr: int = 24000):
    cmd = [_ffmpeg_bin(), "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", out_wav]
    if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("ffmpeg failed to extract audio")

def _read_wav_mono(path: str):
    import wave
    with wave.open(path, "rb") as wf:
        nchan = wf.getnchannels(); sr = wf.getframerate(); nframes = wf.getnframes(); sampwidth = wf.getsampwidth()
        raw = wf.readframes(nframes)
    if sampwidth != 2:
        raise RuntimeError("expected 16-bit PCM WAV")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nchan > 1:
        x = x.reshape(-1, nchan).mean(axis=1)
    return x, sr

def _stft_mag(x: np.ndarray, sr: int, n_fft: int, hop: int):
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = np.hanning(n_fft).astype(np.float32)
    frames = []
    for start in range(0, len(x) - n_fft + 1, hop):
        seg = x[start:start+n_fft] * win
        spec = np.fft.rfft(seg, n=n_fft)
        frames.append(np.abs(spec).astype(np.float32))
    M = np.vstack(frames) if frames else np.zeros((0, n_fft//2+1), dtype=np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0/sr)
    return M, freqs

def _band_energy(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float):
    idx = (freqs >= lo) & (freqs <= hi)
    return mag[:, idx].sum(axis=1) if np.any(idx) else np.zeros((mag.shape[0],), dtype=np.float32)

def _rms_frames(x: np.ndarray, n_fft: int, hop: int):
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    out = []
    for start in range(0, len(x) - n_fft + 1, hop):
        seg = x[start:start+n_fft]
        out.append(float(np.sqrt(np.mean(seg**2) + 1e-12)))
    return np.array(out, dtype=np.float32)

def _spectral_flux(mag: np.ndarray):
    if mag.shape[0] < 2:
        return np.zeros((mag.shape[0],), dtype=np.float32)
    diff = np.maximum(mag[1:] - mag[:-1], 0.0)
    flux = np.sqrt((diff * diff).sum(axis=1))
    return np.concatenate([[0.0], flux.astype(np.float32)])

def _running_quantile(x: np.ndarray, q: float, win: int):
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float32)
    win = max(1, win | 1)  # force odd
    half = win // 2
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = max(0, i - half); e = min(n, i + half + 1)
        out[i] = np.quantile(x[s:e], q)
    return out

def detect_hitmarker_events(input_video: str, cfg: dict) -> Tuple[List[Tuple[float,float,float,float]], float]:
    """
    Returns list of events: (start_sec, end_sec, peak_sec, peak_high_band) and sec_per_frame
    Also writes probe CSV/TXT/VTT to output/details/.
    """
    sr = int(cfg.get("hit_sr", 24000))
    n_fft = int(cfg.get("hit_n_fft", 1024))
    hop = int(cfg.get("hit_hop", 240))

    high_lo = float(cfg.get("hit_high_lo", 3000.0))
    high_hi = float(cfg.get("hit_high_hi", 11000.0))
    low_lo  = float(cfg.get("hit_low_lo", 80.0))
    low_hi  = float(cfg.get("hit_low_hi", 1800.0))

    hi_q       = float(cfg.get("hit_high_quantile", 0.98))
    snr_thresh = float(cfg.get("hit_snr_thresh", 6.0))

    adaptive   = bool(cfg.get("hit_adaptive", True))
    win_sec    = float(cfg.get("hit_win_sec", 1.5))
    loc_hi_q   = float(cfg.get("hit_local_high_q", 0.90))
    loc_snr_q  = float(cfg.get("hit_local_snr_q", 0.90))
    alpha_hi   = float(cfg.get("hit_alpha_high", 1.0))
    alpha_snr  = float(cfg.get("hit_alpha_snr", 1.0))

    use_flux   = bool(cfg.get("hit_use_flux", True))
    flux_q     = float(cfg.get("hit_flux_quantile", 0.98))
    flux_relax_high_q = float(cfg.get("hit_flux_relax_high_q", 0.90))

    min_event_ms    = float(cfg.get("hit_min_event_ms", 25.0))
    refractory_ms   = float(cfg.get("hit_refractory_ms", 60.0))
    merge_gap_sec   = float(cfg.get("hit_merge_gap_sec", 0.18))

    prefix = cfg.get("hit_probe_prefix", "bo7_probe")

    # Extract & read audio via temp wav
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "a.wav")
        cmd_out = details_path(cfg, f"{prefix}_ffmpeg_extract.log")
        try:
            _export_mono_wav(input_video, wav_path, sr=sr)
        except Exception as e:
            with open(cmd_out, "w", encoding="utf-8") as f:
                f.write(str(e))
            raise
        x, sr = _read_wav_mono(wav_path)

    # Frame metrics
    M, freqs = _stft_mag(x, sr, n_fft=n_fft, hop=hop)
    if M.shape[0] == 0:
        return [], hop / float(sr)
    sec_per_frame = hop / float(sr)
    times = np.arange(M.shape[0]) * sec_per_frame

    high = _band_energy(M, freqs, high_lo, high_hi)
    low  = _band_energy(M, freqs, low_lo, low_hi)
    rms  = _rms_frames(x, n_fft, hop)
    snr_db = 20.0 * np.log10((high + 1e-6) / (low + 1e-6))
    flux = _spectral_flux(M)

    hi_glob = np.quantile(high[high>0], hi_q) if np.any(high>0) else 0.0
    snr_glob = snr_thresh

    if adaptive:
        win_frames = int(round(win_sec / sec_per_frame))
        win_frames = max(3, win_frames) | 1  # odd
        hi_loc  = _running_quantile(high,   loc_hi_q, win_frames) * alpha_hi
        snr_loc = _running_quantile(snr_db, loc_snr_q, win_frames) * alpha_snr
    else:
        hi_loc  = np.zeros_like(high)   + hi_glob
        snr_loc = np.zeros_like(snr_db) + snr_glob

    if use_flux:
        flux_thr = np.quantile(flux, flux_q)
    else:
        flux_thr = np.inf

    min_len_frames = max(1, int((min_event_ms/1000.0) / sec_per_frame))
    refractory_frames = max(1, int((refractory_ms/1000.0) / sec_per_frame))

    candidates = []
    i = 0
    while i < len(times):
        hi_gate  = (high[i] >= max(hi_glob, hi_loc[i]))
        snr_gate = (snr_db[i] >= max(snr_glob, snr_loc[i]))
        flux_gate = (use_flux and flux[i] >= flux_thr and
                     (high[i] >= np.quantile(high[high>0], flux_relax_high_q) if np.any(high>0) else False))
        if (hi_gate and snr_gate) or flux_gate:
            j = i
            local_peak = high[i]; local_pt = times[i]
            while j+1 < len(times) and (high[j+1] >= 0.5*high[i]):
                j += 1
                if high[j] > local_peak:
                    local_peak = high[j]; local_pt = times[j]
            if (j - i + 1) >= min_len_frames:
                s = times[i]; e = times[j]
                candidates.append((s, e, local_pt, float(local_peak)))
                i = j + refractory_frames
                continue
        i += 1

    def _merge_events(cands, min_gap_sec):
        if not cands: return []
        cands = sorted(cands, key=lambda z: z[0])
        merged = []
        cur_s, cur_e, cur_peak_t, cur_peak_val = cands[0]
        for s, e, pt, pv in cands[1:]:
            if s - cur_e <= min_gap_sec:
                cur_e = max(cur_e, e)
                if pv > cur_peak_val:
                    cur_peak_t, cur_peak_val = pt, pv
            else:
                merged.append((cur_s, cur_e, cur_peak_t, cur_peak_val))
                cur_s, cur_e, cur_peak_t, cur_peak_val = s, e, pt, pv
        merged.append((cur_s, cur_e, cur_peak_t, cur_peak_val))
        return merged

    events = _merge_events(candidates, merge_gap_sec)

    # Write probes into details/
    frames_csv = details_path(cfg, f"{prefix}_frames.csv")
    with open(frames_csv, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["t_sec","rms","high_band","low_band","snr_db","flux"])
        for k in range(len(times)):
            w.writerow([f"{times[k]:.6f}", f"{rms[k]:.6f}", f"{high[k]:.6f}", f"{low[k]:.6f}",
                        f"{snr_db[k]:.3f}", f"{flux[k]:.6f}"])

    events_csv = details_path(cfg, f"{prefix}_events.csv")
    pre_buf = float(cfg.get("hit_pre_buffer_sec", 0.30))
    post_buf = float(cfg.get("hit_post_buffer_sec", 5.00))
    with open(events_csv, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["start_sec","end_sec","peak_sec","peak_high_band","keep_from","keep_to"])
        for (s,e,pt,pv) in events:
            keep_s = max(0.0, s - pre_buf)
            keep_e = e + post_buf
            w.writerow([f"{s:.6f}", f"{e:.6f}", f"{pt:.6f}", f"{pv:.6f}", f"{keep_s:.6f}", f"{keep_e:.6f}"])

    txt = details_path(cfg, f"{prefix}_transcript.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("# Audio event transcript (adaptive)\n")
        f.write(f"# adaptive={adaptive}, flux={use_flux}\n")
        for (s,e,pt,pv) in events:
            f.write(f"[{fmt_time(s)} --> {fmt_time(e)}] peak@{fmt_time(pt)}  high={pv:.4f}  "
                    f"keep=[{fmt_time(max(0,s-pre_buf))} → {fmt_time(e+post_buf)}]\n")

    vtt = details_path(cfg, f"{prefix}_markers.vtt")
    with open(vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for idx, (s,e,pt,_pv) in enumerate(events, 1):
            def _vtt_ts(t):
                total = float(t)
                hh = int(total // 3600); mm = int((total % 3600) // 60); ss = total % 60
                return f"{hh:02d}:{mm:02d}:{ss:06.3f}".replace(".", ",").replace(",", ".")
            start = _vtt_ts(max(0.0, pt - 0.05)); end = _vtt_ts(pt + 0.05)
            f.write(f"{idx}\n{start} --> {end}\nHit-like peak\n\n")

    return events, sec_per_frame

def build_chained_keep_windows_from_events(
    events: List[Tuple[float,float,float,float]],
    pre_buf: float,
    post_buf: float,
    chain_gap_sec: float,
    duration: float
) -> List[Tuple[float,float]]:
    if not events:
        return []
    events_sorted = sorted(events, key=lambda x: x[0])

    keeps: List[Tuple[float,float]] = []
    cur_last_e = events_sorted[0][1]
    win_s = max(0.0, events_sorted[0][0] - pre_buf)
    win_e = min(duration, cur_last_e + post_buf)

    for (s, e, _pt, _pv) in events_sorted[1:]:
        if (s - cur_last_e) <= chain_gap_sec:
            cur_last_e = max(cur_last_e, e)
            win_e = min(duration, max(win_e, e + post_buf))
        else:
            keeps.append((clamp(win_s, 0.0, duration), clamp(win_e, 0.0, duration)))
            cur_last_e = e
            win_s = max(0.0, s - pre_buf)
            win_e = min(duration, e + post_buf)

    keeps.append((clamp(win_s, 0.0, duration), clamp(win_e, 0.0, duration)))
    return merge_intervals(keeps, min_gap=0.01)
