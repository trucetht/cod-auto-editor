import re
from typing import List, Dict
from .models import Trigger, Asset

def _match(text: str, trigger: Trigger) -> bool:
    t = trigger.match_type.lower().strip()
    phrase = (trigger.phrase or "").strip().lower()
    hay = (text or "").strip().lower()
    if not phrase:
        return False
    if t == "exact":
        return hay == phrase
    if t == "regex":
        try:
            return bool(re.search(phrase, hay))
        except Exception:
            return False
    # default: contains
    return phrase in hay

def find_overlay_events(segments: List[dict],
                        triggers: List[Trigger],
                        assets: Dict[int, Asset],
                        cfg: dict) -> List[dict]:
    """
    Returns list of overlay events:
      { "t": start_time_in_original, "duration": float, "asset": Asset }
    Honors trigger min_cooldown_sec and priority (higher first).
    """
    if not triggers or not assets:
        return []

    # Sort triggers by priority desc
    triggers_sorted = sorted(triggers, key=lambda t: t.priority, reverse=True)
    cooldown_track = {t.id: -1e9 for t in triggers_sorted}

    events: List[dict] = []
    for seg in segments:
        seg_s = float(seg["start"]); seg_e = float(seg["end"])
        text = (seg.get("text") or "").lower()
        for t in triggers_sorted:
            if _match(text, t):
                last_time = cooldown_track.get(t.id, -1e9)
                if (seg_s - last_time) >= float(max(0.0, t.min_cooldown_sec)):
                    asset = assets.get(t.asset_id)
                    if asset:
                        dur = float(asset.default_duration_sec or 2.0)
                        events.append({"t": seg_s, "duration": dur, "asset": asset})
                        cooldown_track[t.id] = seg_s
                        break  # only fire the highest-priority trigger per segment
    return events
