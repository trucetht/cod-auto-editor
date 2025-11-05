from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class Asset:
    id: int
    kind: str
    path: str
    default_duration_sec: float
    position: str
    scale_pct: float
    fade_in_ms: int
    fade_out_ms: int

@dataclass
class Trigger:
    id: int
    phrase: str
    match_type: str  # "exact" | "contains" | "regex"
    asset_id: int
    min_cooldown_sec: float
    priority: int
