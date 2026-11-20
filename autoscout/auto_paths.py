"""
Autonomous paths: where each robot starts, drives and ends during AUTO.

Built from robot_positions.csv (the tracker's per-frame positions, inches,
center-origin, x right, y toward the camera). Robot tracking is the most
reliable output, so this is the scouting view to lean on:

- start / end position (median of the first / last second it is seen),
- the path, resampled every PATH_STEP_S seconds (tracking glitches - jumps
  faster than a robot can drive - are dropped),
- distance driven, how far it got from its start, whether it moved at all,
- how much of AUTO it was visible.

write() saves auto_paths.json, auto_paths.csv and auto_paths.png in the match
folder; path_similarity() compares two paths for consistency across matches.
"""
import csv
import json
import math
import os
from typing import Dict, List, Optional

from autoscout import season

PATH_STEP_S = 0.25          # path sample spacing
MAX_SPEED_IN_S = 100.0      # faster than this between samples = tracking glitch (label jump)
MAX_GAP_CREDIT_S = 0.35     # a gap in tracking allows at most this much driving in one jump
MIN_PIECE_S = 1.5           # path pieces shorter than this between jumps are blips onto another robot
MOVED_MIN_IN = 12.0         # a robot that never gets this far from its start "did not move"
ROBOT_COLORS_BGR = [(0, 165, 255), (255, 200, 0), (180, 0, 255), (0, 220, 120)]


def _median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    return (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])) if n else float("nan")


def extract(pose_rows: List[Dict[str, str]], auto_end_s: Optional[float] = None) -> List[Dict[str, object]]:
    """One entry per robot: {robot, points [[t,x,y]], start, end, distance_in,
    max_from_start_in, moved, visible_pct}."""
    auto_end_s = season.AUTO_END_S if auto_end_s is None else auto_end_s
    robots = []
    for k in range(4):
        raw = []
        for row in pose_rows:
            try:
                t = float(row["timestamp_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if t < 0 or t > auto_end_s:
                continue
            if str(row.get("robot{}_visible".format(k))) != "1":
                continue
            try:
                raw.append((t, float(row["robot{}_x_in".format(k)]), float(row["robot{}_y_in".format(k)])))
            except (KeyError, TypeError, ValueError):
                continue
        # resample: median position in each PATH_STEP_S bin, glitches dropped
        bins: Dict[int, List] = {}
        for t, x, y in raw:
            bins.setdefault(int(t / PATH_STEP_S), []).append((x, y))
        samples = [[round((b + 0.5) * PATH_STEP_S, 2), round(_median([p[0] for p in bins[b]]), 1),
                     round(_median([p[1] for p in bins[b]]), 1)] for b in sorted(bins)]
        # split into pieces wherever the robot would have moved faster than a
        # robot can (label jumped to another robot), then drop short pieces
        # (blips); the first piece is kept when it starts AUTO (start position)
        pieces: List[List[List[float]]] = []
        for smp in samples:
            if pieces:
                pt, px, py = pieces[-1][-1]
                if math.hypot(smp[1] - px, smp[2] - py) <= MAX_SPEED_IN_S * min(max(smp[0] - pt, PATH_STEP_S), MAX_GAP_CREDIT_S):
                    pieces[-1].append(smp)
                    continue
            pieces.append([smp])
        points = []
        for i, piece in enumerate(pieces):
            long_enough = piece[-1][0] - piece[0][0] + PATH_STEP_S >= MIN_PIECE_S
            starts_auto = i == 0 and piece[0][0] <= 2.0
            if long_enough or starts_auto or len(pieces) == 1:
                points.extend(piece)
        entry = {"robot": "R{}".format(k), "points": points, "start": None, "end": None,
                 "distance_in": 0.0, "max_from_start_in": 0.0, "moved": False,
                 "visible_pct": min(100, round(100.0 * len(points) * PATH_STEP_S / max(auto_end_s, 1e-6)))}
        if points:
            t0, t1 = points[0][0], points[-1][0]
            first = [p for p in points if p[0] <= t0 + 1.0]
            last = [p for p in points if p[0] >= t1 - 1.0]
            start = [round(_median([p[1] for p in first]), 1), round(_median([p[2] for p in first]), 1)]
            end = [round(_median([p[1] for p in last]), 1), round(_median([p[2] for p in last]), 1)]
            dist = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(points, points[1:]))
            far = max(math.hypot(p[1] - start[0], p[2] - start[1]) for p in points)
            entry.update(start=start, end=end, distance_in=round(dist, 1), max_from_start_in=round(far, 1),
                         moved=far >= MOVED_MIN_IN, seen_from_s=t0)
        robots.append(entry)
    return robots


def path_similarity(a: List[List[float]], b: List[List[float]], samples: int = 20) -> Optional[float]:
    """Mean distance (inches) between two paths sampled at the same AUTO
    times; lower = more consistent. None when either path is too short."""
    if len(a) < 2 or len(b) < 2:
        return None

    def at(path, t):
        if t <= path[0][0]:
            return path[0][1], path[0][2]
        for p, q in zip(path, path[1:]):
            if p[0] <= t <= q[0]:
                f = (t - p[0]) / max(q[0] - p[0], 1e-6)
                return p[1] + f * (q[1] - p[1]), p[2] + f * (q[2] - p[2])
        return path[-1][1], path[-1][2]
    t0 = max(a[0][0], b[0][0])
    t1 = min(a[-1][0], b[-1][0])
    if t1 <= t0:
        return None
    total = 0.0
    for i in range(samples):
        t = t0 + (t1 - t0) * i / (samples - 1)
        (ax, ay), (bx, by) = at(a, t), at(b, t)
        total += math.hypot(ax - bx, ay - by)
    return round(total / samples, 1)


def load_or_build(match_dir: str) -> Optional[List[Dict[str, object]]]:
    """auto_paths.json of a processed match, built from robot_positions.csv
    for matches processed before auto paths existed."""
    path = os.path.join(match_dir, "auto_paths.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("robots")
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(match_dir, "robot_positions.csv"), newline="", encoding="utf-8") as fh:
            return extract(list(csv.DictReader(fh)))
    except OSError:
        return None


def write(cv2, np, output_dir: str, pose_rows: List[Dict[str, str]], size: int = 720) -> List[Dict[str, object]]:
    robots = extract(pose_rows)
    with open(os.path.join(output_dir, "auto_paths.json"), "w", encoding="utf-8") as fh:
        json.dump({"auto_end_s": season.AUTO_END_S, "robots": robots}, fh, indent=1)
    with open(os.path.join(output_dir, "auto_paths.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["robot", "match_time_s", "x_in", "y_in"])
        for r in robots:
            for t, x, y in r["points"]:
                w.writerow([r["robot"], t, x, y])
    # picture: each robot's path on the field, start dot, end arrow
    from autoscout.analytics import _field_canvas, _to_px
    canvas = _field_canvas(cv2, np, size)
    for i, r in enumerate(robots):
        pts = [_to_px(x, y, size) for _t, x, y in r["points"]]
        if len(pts) < 1:
            continue
        col = ROBOT_COLORS_BGR[i % len(ROBOT_COLORS_BGR)]
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, col, 3, cv2.LINE_AA)
            cv2.arrowedLine(canvas, pts[max(0, len(pts) - 3)], pts[-1], col, 3, cv2.LINE_AA, tipLength=0.6)
        cv2.circle(canvas, pts[0], 9, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, pts[0], 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, r["robot"], (pts[0][0] + 11, pts[0][1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "AUTO paths (dot = start, arrow = end)", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "auto_paths.png"), canvas)
    return robots
