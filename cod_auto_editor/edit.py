from typing import List, Tuple
from .intervals import merge_intervals, subtract_intervals, clamp

def map_time_to_kept_timeline(t_original: float, keep_segments: List[Tuple[float,float]]) -> float | None:
    acc = 0.0
    for (s, e) in keep_segments:
        if t_original < s:
            return None
        if s <= t_original <= e:
            return acc + (t_original - s)
        acc += (e - s)
    return None

def parse_position(position: str, W: int, H: int, w: int, h: int) -> Tuple[int, int]:
    m = 20
    if position == "top-left": return (m, m)
    if position == "top-right": return (W - w - m, m)
    if position == "bottom-left": return (m, H - h - m)
    if position == "bottom-right": return (W - w - m, H - h - m)
    return ((W - w)//2, (H - h)//2)

def build_edit_pieces(duration: float,
                      filler_cuts: List[Tuple[float,float]],
                      silence_cuts: List[Tuple[float,float]],
                      downtime_short: List[Tuple[float,float]],
                      downtime_long: List[Tuple[float,float]]) -> List[Tuple[float,float,str]]:
    hard_cuts = merge_intervals(filler_cuts + silence_cuts + downtime_long, min_gap=0.05)
    ramp_segments = subtract_intervals(merge_intervals(downtime_short, 0.05), hard_cuts)

    boundaries = {0.0, duration}
    for s,e in hard_cuts + ramp_segments:
        boundaries.add(clamp(s, 0.0, duration))
        boundaries.add(clamp(e, 0.0, duration))
    bounds = sorted(boundaries)

    pieces = []
    for i in range(len(bounds)-1):
        s, e = bounds[i], bounds[i+1]
        if e - s <= 1e-6:
            continue
        label = 'keep'
        for cs,ce in hard_cuts:
            if s >= cs and e <= ce:
                label = 'cut'
                break
        if label != 'cut':
            for rs,re in ramp_segments:
                if s >= rs and e <= re:
                    label = 'ramp'
                    break
        pieces.append((s,e,label))
    return pieces

def build_keep_only_pieces(duration: float, keep_ranges: List[Tuple[float,float]]) -> List[Tuple[float,float,str]]:
    keeps = merge_intervals([(clamp(s,0.0,duration), clamp(e,0.0,duration)) for (s,e) in keep_ranges], min_gap=0.05)
    pieces = []
    for (s,e) in keeps:
        if e > s:
            pieces.append((s,e,'keep'))
    return pieces
