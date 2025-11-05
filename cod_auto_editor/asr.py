import os
import torch
from typing import List, Optional
from faster_whisper import WhisperModel

# Optional (stable-ts)
_HAS_STABLE = False
try:
    import stable_whisper as swhisper  # pip install stable-ts
    _HAS_STABLE = True
except Exception:
    _HAS_STABLE = False

def _fw_pick_device_and_type(cfg: dict):
    """Yield (device, compute_type) candidates with fallbacks."""
    order = cfg.get("whisper_device_order") or ["metal", "cuda", "cpu"]
    types_cfg = cfg.get("whisper_compute_types") or {}
    def_ctype = {"metal": "float16", "cuda": "float16", "cpu": "int8_float32"}
    for k,v in def_ctype.items():
        types_cfg.setdefault(k, v)

    avail = {
        "metal": bool(os.environ.get("CT2_USE_METAL", "1") != "0"),
        "cuda":  torch.cuda.is_available(),
        "cpu":   True,
    }
    for dev in order:
        if avail.get(dev, False):
            yield dev, types_cfg.get(dev, def_ctype[dev])

def _transcribe_faster_whisper(video_path: str, cfg: dict, purpose: str) -> List[dict]:
    model_name = cfg.get("whisper_model", "large-v3")
    fast = bool(cfg.get("decoding_fast", True))
    decode_kwargs = dict(
        beam_size=1 if fast else 5,
        best_of=1 if fast else 5,
        temperature=0.0,
        condition_on_previous_text=False,
    )

    last_err = None
    model = None
    chosen = None
    for dev, ctype in _fw_pick_device_and_type(cfg):
        try:
            print(f"[ASR] faster-whisper | trying device={dev} ctype={ctype} model={model_name}")
            model = WhisperModel(model_name, device=dev, compute_type=ctype)
            chosen = (dev, ctype)
            break
        except Exception as e:
            last_err = e
            print(f"[ASR] faster-whisper init failed on device={dev} ({ctype}): {e}")
            model = None
            continue
    if model is None:
        raise RuntimeError(f"faster-whisper could not initialize on any device (last error: {last_err})")

    print(f"[ASR] faster-whisper using device={chosen[0]} compute_type={chosen[1]} model={model_name}")
    segments_iter, _info = model.transcribe(
        video_path,
        language="en",
        vad_filter=bool(cfg.get("use_vad", True)),
        word_timestamps=bool(cfg.get("word_timestamps", True)),
        **decode_kwargs
    )
    out = []
    for seg in segments_iter:
        words = []
        if getattr(seg, "words", None):
            for w in seg.words:
                if not w.word:
                    continue
                words.append({
                    "word": w.word.strip(),
                    "start": float(w.start) if w.start is not None else float(seg.start),
                    "end": float(w.end) if w.end is not None else float(seg.end),
                })
        out.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": (seg.text or "").strip(),
            "words": words,
        })
    print(f"[ASR] faster-whisper produced {len(out)} segments ({purpose})")
    return out

def transcribe_audio(video_path: str, cfg: dict, purpose: str) -> List[dict]:
    try_stable = bool(cfg.get("use_stable_ts", False)) and _HAS_STABLE
    if try_stable:
        try:
            print("[ASR] Using stable-ts (experimental).")
            model_name = cfg.get("whisper_model", "large-v3")
            model = swhisper.load_model(model_name, device=cfg.get("force_torch_device","cpu"))
            result = model.transcribe(video_path, word_timestamps=True)
            out = []
            for seg in result["segments"]:
                words = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])} for w in seg["words"]]
                out.append({"start": float(seg["start"]), "end": float(seg["end"]),
                            "text": seg["text"].strip(), "words": words})
            print(f"[ASR] stable-ts produced {len(out)} segments ({purpose})")
            return out
        except Exception as e:
            print(f"[ASR] stable-ts failed ({e.__class__.__name__}): {e}")
            print("[ASR] Falling back to faster-whisper…")

    return _transcribe_faster_whisper(video_path, cfg, purpose)
