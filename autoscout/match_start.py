"""
Match start detection from robot motion (no broadcast clock).

The on-screen clock is not reliable (it can start seconds after the robots
do). This scans the start of the video for the first sign that autonomous
has begun, whichever comes first:

- a robot that had been parked (still for STABLE_S) drives away and keeps
  going, or
- POLLEN flies into a HIVE box (ball-sized moving yellow blob inside a HIVE
  box in two consecutive samples) - robots often shoot from their starting
  position before driving.

Neither cue is fully reliable on event streams (robots may shoot without
moving; people walking behind the clear HIVE can look like flying POLLEN
before the match), so the results are candidates to confirm by eye -
tools/find_match_start.py prints them and saves frames around each.

People walking on the field before the match are not still, so they never
become anchors. A person standing still and then walking away can still
trigger a false start; check the reported time.
"""
import math
from typing import Dict, List, Optional


class MatchStartDetector:
    SAMPLE_S = 0.2          # scan step
    STILL_IN = 4.0          # an anchor's blob stays within this many inches
    STABLE_S = 2.0          # ...for this long before it counts as a parked robot
    DEPART_IN = 8.0         # it must then move at least this far away
    KEEP_MOVING_S = 1.0     # and not come back within this long
    MAX_SCAN_S = 45.0
    BALL_MIN_PX = 3         # ball-sized moving yellow blob (px area at 640 wide)
    BALL_MAX_PX = 200

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np

    def _blob_positions(self, tracker, frame) -> List[tuple]:
        cv2 = self.cv2
        contours, _centers = tracker._foreground_contours(frame)
        out = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            bottom = float(c[:, 0, 1].max())
            fx, fy = tracker._image_to_field(M["m10"] / M["m00"], M["m01"] / M["m00"], bottom_v=bottom)
            if -6 <= fx <= 150 and -6 <= fy <= 150:
                out.append((fx, fy))
        return out

    def detect(self, video_path: str, tracker, max_scan_s: Optional[float] = None) -> Dict[str, object]:
        cv2 = self.cv2
        self._prev_frame = None
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_scan_s = self.MAX_SCAN_S if max_scan_s is None else max_scan_s
        step = max(1, int(round(self.SAMPLE_S * fps)))
        anchors = []   # dicts: pos, since, last_seen, gone_since, stable
        departures = []
        frame_idx = 0
        launch_hits = 0
        last_launch_t = None
        first_launch = None
        while frame_idx < min(total, int(max_scan_s * fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            launch = self._ball_in_hive(tracker, frame)
            blobs = self._blob_positions(tracker, frame)
            used = set()
            for a in anchors:
                near = [k for k, b in enumerate(blobs)
                        if k not in used and math.dist(b, a["pos"]) <= self.STILL_IN]
                if near:
                    used.add(near[0])
                    a["last_seen"] = t
                    a["gone_since"] = None
                    if not a["stable"] and t - a["since"] >= self.STABLE_S:
                        a["stable"] = True
                else:
                    if a["gone_since"] is None:
                        a["gone_since"] = t
            # Launch cue only counts while everything on the field is a parked
            # robot (no people walking around, e.g. loading the HIVE).
            settled = bool(blobs) and all(
                any(a["stable"] and math.dist(bp, a["pos"]) <= self.STILL_IN for a in anchors)
                for bp in blobs)
            if launch and settled:
                launch_hits = launch_hits + 1 if last_launch_t is not None and t - last_launch_t <= self.SAMPLE_S * 1.5 else 1
                last_launch_t = t
                if launch_hits >= 2 and first_launch is None:
                    first_launch = t - self.SAMPLE_S
            for k, b in enumerate(blobs):
                if k not in used:
                    anchors.append({"pos": b, "since": t, "last_seen": t,
                                    "gone_since": None, "stable": False})
            # a stable anchor whose robot left and has not come back
            for a in anchors:
                if (a["stable"] and a["gone_since"] is not None
                        and t - a["gone_since"] >= self.KEEP_MOVING_S
                        and not a.get("reported")):
                    moved = [b for b in blobs if self.DEPART_IN <= math.dist(b, a["pos"]) <= 60]
                    a["reported"] = True
                    departures.append({"time_s": a["last_seen"], "pos": a["pos"], "moved_blob": bool(moved)})
            # forget unstable anchors that vanished (people walking)
            anchors = [a for a in anchors
                       if a["stable"] or a["gone_since"] is None or t - a["gone_since"] < 0.6]
            if departures and first_launch is not None:
                break
            frame_idx += step
        cap.release()
        candidates = []
        if departures:
            candidates.append(("robot started driving", min(d["time_s"] for d in departures)))
        if first_launch is not None:
            candidates.append(("POLLEN moving in a HIVE box", first_launch))
        if not candidates:
            return {"start_s": None, "reason": None, "departures": departures}
        reason, start = min(candidates, key=lambda c: c[1])
        return {"start_s": max(0.0, start), "reason": reason, "candidates": candidates,
                "departures": departures}

    def _ball_in_hive(self, tracker, frame) -> bool:
        """Ball-sized yellow blob inside a HIVE box that moved since the
        previous sample (frame-to-frame difference, so balls already resting
        in a CELL do not count)."""
        cv2, np = self.cv2, self.np
        prev = getattr(self, "_prev_frame", None)
        self._prev_frame = frame
        if not tracker.hive_boxes or prev is None:
            return False
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color = None
        for lo, hi in tracker.BALL_HSV_RANGES:
            part = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            color = part if color is None else cv2.bitwise_or(color, part)
        diff = cv2.cvtColor(cv2.absdiff(frame, prev), cv2.COLOR_BGR2GRAY)
        _, moving = cv2.threshold(diff, tracker.BALL_MOTION_THRESH, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(color, moving)
        scale = (frame.shape[1] / 640.0) ** 2
        for poly in tracker.hive_boxes.values():
            box_mask = np.zeros_like(mask)
            cv2.fillPoly(box_mask, [poly], 255)
            cnts, _ = cv2.findContours(cv2.bitwise_and(mask, box_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = max(float(cv2.contourArea(c)), float(len(c)))
                if self.BALL_MIN_PX * scale <= area <= self.BALL_MAX_PX * scale:
                    return True
        return False
