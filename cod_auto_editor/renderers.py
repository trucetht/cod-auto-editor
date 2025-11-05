import os, time, shlex, subprocess
import numpy as np
from typing import List, Tuple, Dict
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx import all as vfx
from .models import Asset
from .edit import map_time_to_kept_timeline, parse_position

def render_with_moviepy(video_path: str, cfg: dict,
                        pieces: List[Tuple[float,float,str]],
                        overlays: List[dict]) -> str:
    clip = VideoFileClip(video_path)
    W, H = clip.w, clip.h
    transition_sec = float(cfg.get("transition_ms", 180)) / 1000.0
    ramp_factor = float(cfg.get("speed_ramp_factor", 2.0))

    stitched = []
    keep_for_mapping: List[Tuple[float,float]] = []
    for (s,e,label) in pieces:
        if label == 'cut':
            continue
        sub = clip.subclip(s,e)
        if label == 'ramp':
            sub = vfx.speedx(sub, ramp_factor)
        else:
            keep_for_mapping.append((s,e))
        stitched.append(sub)

    if not stitched:
        clip.close()
        raise RuntimeError("Resulting edit removed all content.")

    if len(stitched) == 1 or transition_sec <= 0:
        base = stitched[0]
    else:
        base = concatenate_videoclips(stitched, method="compose", padding=-transition_sec)

    overlay_clips = []
    if overlays:
        safety = float(cfg.get("safe_margin_sec", 0.25))
        for ev in overlays:
            t0 = float(ev["t"]); dur = float(ev["duration"])
            asset: Asset = ev["asset"]
            t_new = map_time_to_kept_timeline(t0, keep_for_mapping)
            if t_new is None:
                continue
            t_new = max(0.0, t_new + safety)

            if asset.kind == "image" or asset.path.lower().endswith(".gif"):
                img = ImageClip(asset.path)
                scale = float(asset.scale_pct) / 100.0
                new_w = max(1, int(W * scale))
                img = img.resize(width=new_w)
                iw, ih = img.w, img.h
                pos = parse_position(asset.position, W, H, iw, ih)
                img = img.set_start(t_new).set_duration(dur).set_position(pos)
                if asset.fade_in_ms > 0:  img = img.fadein(asset.fade_in_ms / 1000.0)
                if asset.fade_out_ms > 0: img = img.fadeout(asset.fade_out_ms / 1000.0)
                overlay_clips.append(img)

    final = CompositeVideoClip([base, *overlay_clips], size=(W, H))

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"final_{int(time.time())}.mp4")
    final.write_videofile(
        out_path,
        fps=int(cfg.get("target_fps", 60)),
        codec=str(cfg.get("codec", "libx264")),
        audio_codec=str(cfg.get("audio_codec", "aac")),
        bitrate=str(cfg.get("bitrate", "10M")),
        preset=str(cfg.get("preset", "medium")),
        threads=os.cpu_count() or 4
    )
    clip.close()
    final.close()
    return out_path

def render_with_ffmpeg_hdr(video_path: str, cfg: dict,
                           pieces: List[Tuple[float,float,str]]) -> str:
    def _ffmpeg_bin_cli() -> str:
        return "ffmpeg"

    vf_parts = []
    af_parts = []

    ramp = float(cfg.get("speed_ramp_factor", 2.0))
    if ramp > 2.0:
        reps = int(np.ceil(np.log(ramp)/np.log(2)))
        atempo_chain = ",".join(["atempo=2.0"] * reps)
    else:
        atempo_chain = f"atempo={ramp}"

    idx = 0
    for (s,e,label) in pieces:
        if label == "cut":
            continue
        vlab = f"[v{idx}]"; alab = f"[a{idx}]"
        vf = f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS"
        af = f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS"
        if label == "ramp":
            vf += f",setpts=PTS/{ramp}"
            af += f",{atempo_chain}"
        vf += vlab
        af += alab
        vf_parts.append(vf)
        af_parts.append(af)
        idx += 1

    if idx == 0:
        raise RuntimeError("Resulting edit removed all content.")

    pads_interleaved = "".join(f"[v{i}][a{i}]" for i in range(idx))
    filtergraph = ";".join(vf_parts + af_parts + [pads_interleaved + f"concat=n={idx}:v=1:a=1[vout][aout]"])

    os.makedirs("output", exist_ok=True)
    ts = int(time.time())
    out_path = os.path.join("output", f"final_{ts}.mp4")

    color_primaries = cfg.get("color_primaries", "bt2020")
    colorspace      = cfg.get("colorspace", "bt2020nc")
    color_trc       = cfg.get("color_trc", "arib-std-b67")
    audio_bitrate   = cfg.get("audio_bitrate","320k")

    pix_fmt = cfg.get("pix_fmt_10bit", "p010le")
    tag = cfg.get("video_tag", "hvc1")
    bitrate = cfg.get("bitrate","60M")
    maxrate = cfg.get("maxrate","120M")
    bufsize = cfg.get("bufsize","240M")

    cmd = [
        _ffmpeg_bin_cli(), "-y",
        "-i", video_path,
        "-filter_complex", filtergraph,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "hevc_videotoolbox",
        "-profile:v", "main10",
        "-pix_fmt", pix_fmt,
        "-tag:v", tag,
        "-b:v", bitrate, "-maxrate", maxrate, "-bufsize", bufsize,
        "-color_primaries", color_primaries,
        "-colorspace", colorspace,
        "-color_trc", color_trc,
        "-c:a", "aac", "-b:a", audio_bitrate,
        out_path
    ]

    print("FFmpeg HDR command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed (see output above)")
    return out_path
