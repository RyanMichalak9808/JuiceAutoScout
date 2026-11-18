"""
HIVE tip detection (BIOBUZZ).

Each HIVE is a bi-stable seesaw with two CELLS. Seen from the audience-side
camera, the alliance-colored HIVE pixels inside its calibrated box sit at one
of two heights depending on which CELL faces up, and jump between them when
the HIVE tips. The detector tracks the vertical centroid of those pixels and
reports a tip whenever the centroid moves to a new level and stays there.

Requires hive_boxes_px in field_corners.json (draw them with
tools/calibrate.py, key H).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from autoscout import season


def measure_hive_level(cv2, np, frame, box, ranges, min_color_frac: float = 0.03) -> Optional[float]:
    """Vertical centroid (0 = top, 1 = bottom of box) of the alliance color
    inside box = (x0, y0, x1, y1), or None if too little of it is visible."""
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in ranges:
        part = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    ys, _xs = np.nonzero(mask)
    if len(ys) < min_color_frac * mask.size:
        return None
    return float(ys.mean()) / max(y1 - y0 - 1, 1)


def hive_box_bounds(poly, w: int, h: int):
    pts = poly.reshape(-1, 2)
    x0 = max(0, int(pts[:, 0].min()))
    x1 = min(w, int(pts[:, 0].max()) + 1)
    y0 = max(0, int(pts[:, 1].min()))
    y1 = min(h, int(pts[:, 1].max()) + 1)
    return x0, y0, x1, y1


@dataclass
class TipEvent:
    alliance: str
    tip_number: int          # running count for this alliance
    match_time_s: float      # when the move to the new level started
    frame_num: int
    level_before: float      # centroid height in the box (0 = top, 1 = bottom)
    level_after: float


class HiveTipDetector:
    # A tip is a centroid move of at least this fraction of the box height...
    MIN_SHIFT_FRAC = 0.10
    # ...that holds for this long (seconds). Filters robots/people briefly
    # covering the HIVE and the swing itself.
    CONFIRM_S = 1.0
    # Ignore frames where fewer than this fraction of the box pixels have the
    # alliance color (HIVE hidden by something).
    MIN_COLOR_FRAC = 0.03
    # Samples during CONFIRM_S must agree with each other within this
    # fraction of the box height (median absolute deviation).
    MAX_SPREAD_FRAC = 0.05

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self._boxes: Dict[str, tuple] = {}
        self._level: Dict[str, Optional[float]] = {}
        self._pending: Dict[str, list] = {}
        self._warmup: Dict[str, list] = {}
        self.counts: Dict[str, int] = {}
        self.events: List[TipEvent] = []
        self.last_level: Dict[str, Optional[float]] = {}
        self._ready = False

    def setup(self, tracker) -> None:
        if not tracker.hive_boxes:
            print("[WARN] HIVE tip detection disabled: no hive_boxes_px in calibration "
                  "(draw them with tools/calibrate.py, key H).")
            return
        h, w = tracker._field_mask.shape[:2]
        for color, poly in tracker.hive_boxes.items():
            if color not in season.HIVE_HSV_RANGES:
                continue
            x0, y0, x1, y1 = hive_box_bounds(poly, w, h)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            self._boxes[color] = (x0, y0, x1, y1)
            self._level[color] = None
            self._pending[color] = []
            self._warmup[color] = []
            self.counts[color] = 0
            self.last_level[color] = None
        self._ready = bool(self._boxes)
        if self._ready:
            print("[INFO] HIVE tip detection ready: {}".format(", ".join(sorted(self._boxes))))

    def measure(self, frame, color: str) -> Optional[float]:
        """Centroid height (0..1 of box) of the alliance color, or None."""
        return measure_hive_level(self.cv2, self.np, frame, self._boxes[color],
                                  season.HIVE_HSV_RANGES[color], self.MIN_COLOR_FRAC)

    def state(self, match_time_s: float) -> Dict[str, tuple]:
        """{color: (last measured level or None, is_moving)} for the tracker."""
        return {color: (self.last_level.get(color), self.is_moving(color, match_time_s))
                for color in self._boxes}

    def update(self, frame, frame_num: int, match_time_s: float) -> List[TipEvent]:
        if not self._ready:
            return []
        np = self.np
        events = []
        for color in self._boxes:
            level = self.measure(frame, color)
            if level is None:
                continue
            self.last_level[color] = level
            sample = (match_time_s, frame_num, level)
            if self._level[color] is None:
                warm = self._warmup[color]
                warm.append(sample)
                if warm[-1][0] - warm[0][0] >= self.CONFIRM_S:
                    self._level[color] = float(np.median([s[2] for s in warm]))
                continue
            current = self._level[color]
            pending = self._pending[color]
            if abs(level - current) < self.MIN_SHIFT_FRAC:
                pending.clear()
                continue
            pending.append(sample)
            if pending[-1][0] - pending[0][0] < self.CONFIRM_S:
                continue
            values = np.array([s[2] for s in pending])
            new_level = float(np.median(values))
            spread = float(np.median(np.abs(values - new_level)))
            if abs(new_level - current) < self.MIN_SHIFT_FRAC or spread > self.MAX_SPREAD_FRAC:
                # Still moving / noisy: keep only the recent part and wait.
                del pending[:len(pending) // 2]
                continue
            self.counts[color] += 1
            event = TipEvent(
                alliance=color,
                tip_number=self.counts[color],
                match_time_s=pending[0][0],
                frame_num=pending[0][1],
                level_before=current,
                level_after=new_level,
            )
            self.events.append(event)
            events.append(event)
            self._level[color] = new_level
            pending.clear()
        return events

    def is_moving(self, color: str, match_time_s: float, window_s: float = 2.0) -> bool:
        """True while this HIVE's level is changing (a tip in progress) or
        within window_s after the start of its last confirmed tip."""
        if self._pending.get(color):
            return True
        for event in reversed(self.events):
            if event.alliance == color:
                return 0.0 <= match_time_s - event.match_time_s <= window_s + self.CONFIRM_S
        return False

    def draw_debug(self, dbg) -> None:
        if not self._ready:
            return
        cv2 = self.cv2
        for color, (x0, y0, x1, y1) in self._boxes.items():
            cv_color = (255, 120, 40) if color == "blue" else (40, 40, 255)
            level = self.last_level.get(color)
            if level is not None:
                y = int(round(y0 + level * (y1 - y0 - 1)))
                cv2.line(dbg, (x0, y), (x1, y), cv_color, 1)
            label = "{} tips: {}".format(color, self.counts.get(color, 0))
            ty = y1 + 14
            cv2.putText(dbg, label, (x0 + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(dbg, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
