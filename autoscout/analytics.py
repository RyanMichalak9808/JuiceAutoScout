"""
Per-match analytics written at the end of a run.

Built to be useful with imperfect (~70%) detection: everything here is a
per-match aggregate meant to be compared ACROSS matches (5-6 qualification
matches per team), where noise averages out.

Outputs (in the run's output folder):
- alliance_summary.csv  per alliance: tips, auto tips, tips in the final 60 s,
  last tip time, median / mean time between tips (cycle cadence)
- robot_summary.csv     per robot track: distance, % of time moving, longest
  stall, field coverage, which half it stayed in, shots made / missed;
  red-flag notes (long stall, never left one half, barely moved)
- shot_heatmap.png      shooter positions on the BIOBUZZ field map, made in
  green / missed in red, with a density glow
- resource_density.csv / resource_density.png  where loose POLLEN (yellow)
  and red / blue NECTAR sit on the field over the match

Robot IDs are tracker labels (R0-R3), not team numbers: map them to teams
(e.g. from the start positions) before aggregating across matches.
"""
import csv
import math
import os
import statistics
from typing import Dict, List

from autoscout import season

FIELD_IN = 144.0

# Loose game pieces on the floor (OpenCV HSV). POLLEN uses the season ball
# color; NECTAR is red / blue (also the alliance colors, so only small,
# round, ball-sized blobs on the floor count).
NECTAR_HSV = {
    "red_nectar": [((0, 120, 80), (8, 255, 255)), ((172, 120, 80), (179, 255, 255))],
    "blue_nectar": [((100, 120, 60), (125, 255, 255))],
}
RESOURCE_SAMPLE_S = 2.0
STALL_SPEED_IN_S = 3.0        # slower than this counts as not moving
STALL_FLAG_S = 30.0           # a stall this long is a red flag
ENDGAME_S = 60.0              # final 60 s (FLOWERS unlock for NECTAR)
LATE_S = 15.0                 # tips in the final 15 s = full-match HIVE cycler


# ── tips ────────────────────────────────────────────────────────────────────
def alliance_summary(tip_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    end = season.MATCH_END_S
    out = []
    for alliance in ("red", "blue"):
        times = sorted(float(r["match_time_s"]) for r in tip_rows
                       if r["alliance"] == alliance and r["phase"] in ("auto", "transition", "teleop"))
        gaps = [b - a for a, b in zip(times, times[1:])]
        out.append({
            "alliance": alliance,
            "tips": len(times),
            "auto_tips": sum(1 for t in times if t < season.AUTO_END_S),
            "teleop_tips": sum(1 for t in times if t >= season.AUTO_END_S),
            "endgame_tips_final_60s": sum(1 for t in times if t >= end - ENDGAME_S),
            "tips_final_15s": sum(1 for t in times if t >= end - LATE_S),
            "first_tip_s": "{:.1f}".format(times[0]) if times else "",
            "last_tip_s": "{:.1f}".format(times[-1]) if times else "",
            "median_s_between_tips": "{:.1f}".format(statistics.median(gaps)) if gaps else "",
            "mean_s_between_tips": "{:.1f}".format(statistics.mean(gaps)) if gaps else "",
            "notes": _tip_notes(times, end),
        })
    return out


def _tip_notes(times, end) -> str:
    notes = []
    if any(t < season.AUTO_END_S for t in times):
        notes.append("scores in AUTO")
    teleop = [t for t in times if t >= season.AUTO_END_S]
    endgame = [t for t in teleop if t >= end - ENDGAME_S]
    early = [t for t in teleop if t < end - ENDGAME_S]
    if len(early) >= 2 and not endgame:
        notes.append("stopped tipping in the final 60 s (FLOWER / Garden pivot?)")
    if any(t >= end - LATE_S for t in times):
        notes.append("still tipping in the final 15 s (full-match HIVE cycler)")
    return "; ".join(notes)


# ── robots ──────────────────────────────────────────────────────────────────
def robot_summary(pose_rows: List[Dict[str, str]], shot_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for rid in range(4):
        pts = []
        for r in pose_rows:
            t = float(r["timestamp_s"])
            if r.get("robot{}_visible".format(rid)) == "1" and 0.0 <= t <= season.MATCH_END_S:
                pts.append((float(r["timestamp_s"]), float(r["robot{}_x_in".format(rid)]),
                            float(r["robot{}_y_in".format(rid)])))
        made = sum(1 for s in shot_rows if s["robot"] == "R{}".format(rid) and s["result"] == "made")
        missed = sum(1 for s in shot_rows if s["robot"] == "R{}".format(rid) and s["result"] == "missed")
        row = {"robot": "R{}".format(rid), "seen_s": 0.0, "distance_in": 0.0, "moving_pct": "",
               "longest_stall_s": 0.0, "field_coverage_pct": "", "time_left_half_pct": "",
               "shots_made": made, "shots_missed": missed, "red_flags": ""}
        if len(pts) < 2:
            row["red_flags"] = "barely seen"
            out.append(row)
            continue
        dist = moving = seen = 0.0
        stall = longest = 0.0
        for (t0, x0, y0), (t1, x1, y1) in zip(pts, pts[1:]):
            dt = t1 - t0
            if dt <= 0 or dt > 1.0:          # hidden in between: don't count
                stall = 0.0
                continue
            d = math.hypot(x1 - x0, y1 - y0)
            seen += dt
            if d / dt > 150.0:               # tracking jump, not motion
                continue
            dist += d
            if d / dt >= STALL_SPEED_IN_S:
                moving += dt
                stall = 0.0
            else:
                stall += dt
                longest = max(longest, stall)
        cells = {(min(5, max(0, int((x + 72) // 24))), min(5, max(0, int((y + 72) // 24)))) for _t, x, y in pts}
        left = sum(1 for _t, x, _y in pts if x < 0) / len(pts)
        flags = []
        if longest >= STALL_FLAG_S:
            flags.append("stalled {:.0f} s".format(longest))
        if left > 0.95 or left < 0.05:
            flags.append("never left one half")
        if seen > 30 and dist / seen < 4.0:
            flags.append("barely moved")
        row.update({
            "seen_s": round(seen, 1), "distance_in": round(dist), "moving_pct": round(100 * moving / max(seen, 1e-6)),
            "longest_stall_s": round(longest, 1), "field_coverage_pct": round(100 * len(cells) / 36.0),
            "time_left_half_pct": round(100 * left), "red_flags": "; ".join(flags),
        })
        out.append(row)
    return out


# ── loose game pieces ───────────────────────────────────────────────────────
class ResourceSampler:
    """Every RESOURCE_SAMPLE_S, finds loose POLLEN / NECTAR on the floor and
    stores their field positions."""

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self.samples: List[tuple] = []     # (match_time_s, kind, x_in, y_in)
        self._next_t = 0.0

    def maybe_sample(self, frame, match_time_s: float, tracker) -> None:
        if match_time_s < self._next_t or tracker._field_mask is None or tracker._H_2d is None:
            return
        self._next_t = match_time_s + RESOURCE_SAMPLE_S
        cv2, np = self.cv2, self.np
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        floor = tracker._field_mask.copy()
        for poly in getattr(tracker, "flower_polys", []):
            cv2.fillPoly(floor, [poly], 0)
        person = getattr(tracker, "_person_mask", None)
        if person is not None:
            floor = cv2.bitwise_and(floor, cv2.bitwise_not(person))
        # robots (red / blue bumpers): large foreground blobs, slightly grown
        fg = getattr(tracker, "_last_fg", None)
        if fg is not None:
            robots = np.zeros_like(fg)
            min_area = 0.3 * tracker._robot_max_px * tracker._robot_max_px * tracker._single_area_ratio
            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) >= min_area:
                    cv2.drawContours(robots, [c], -1, 255, -1)
            robots = cv2.dilate(robots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            floor = cv2.bitwise_and(floor, cv2.bitwise_not(robots))
        ranges = {"pollen": list(season.BALL_HSV_RANGES)}
        ranges.update(NECTAR_HSV)
        scale = (frame.shape[1] / 640.0) ** 2
        for kind, rr in ranges.items():
            mask = None
            for lo, hi in rr:
                part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask = part if mask is None else cv2.bitwise_or(mask, part)
            mask = cv2.bitwise_and(mask, floor)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if not (6 * scale <= area <= 160 * scale):
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if max(w, h) > 2.2 * min(w, h) + 2:
                    continue                      # tape lines, robot parts
                fx, fy = tracker._raw_image_to_field(x + w / 2.0, y + h - 1.0)
                if 0 <= fx <= FIELD_IN and 0 <= fy <= FIELD_IN:
                    self.samples.append((match_time_s, kind, fx - 72.0, fy - 72.0))


# ── images ──────────────────────────────────────────────────────────────────
def _field_canvas(cv2, np, size=720):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), season.FIELD_IMAGE)
    img = cv2.imread(path)
    if img is None:
        img = np.full((size, size, 3), 200, np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return (img * 0.8).astype(np.uint8)


def _to_px(x_in, y_in, size):
    return int(round((x_in + 72.0) / FIELD_IN * (size - 1))), int(round((y_in + 72.0) / FIELD_IN * (size - 1)))


def _glow(cv2, np, canvas, pts, color, size, sigma_frac=0.035, strength=1.0):
    if not pts:
        return canvas
    heat = np.zeros((size, size), np.float32)
    for p in pts:
        x, y = _to_px(p[0], p[1], size)
        if 0 <= x < size and 0 <= y < size:
            heat[y, x] += 1.0
    k = int(size * sigma_frac) * 6 + 1
    heat = cv2.GaussianBlur(heat, (k, k), size * sigma_frac)
    if heat.max() > 0:
        heat /= heat.max()
    layer = np.zeros_like(canvas, np.float32)
    layer[:] = color
    a = (heat * 0.75 * strength)[:, :, None]
    return (canvas * (1 - a) + layer * a).astype(np.uint8)


def write_shot_heatmap(cv2, np, path, shot_rows, size=720):
    canvas = _field_canvas(cv2, np, size)
    made = [(float(s["shooter_x_in"]), float(s["shooter_y_in"])) for s in shot_rows
            if s["result"] == "made" and s["shooter_x_in"] != ""]
    missed = [(float(s["shooter_x_in"]), float(s["shooter_y_in"])) for s in shot_rows
              if s["result"] == "missed" and s["shooter_x_in"] != ""]
    canvas = _glow(cv2, np, canvas, made + missed, (0, 200, 255), size)
    for pts, col in ((missed, (40, 40, 255)), (made, (60, 230, 60))):
        for p in pts:
            cv2.circle(canvas, _to_px(p[0], p[1], size), 6, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, _to_px(p[0], p[1], size), 6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Shot locations: made {} (green) / missed {} (red)".format(len(made), len(missed)),
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def write_resource_density(cv2, np, png_path, csv_path, samples, size=720):
    with open(csv_path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["match_time_s", "kind", "x_in", "y_in"])
        for t, kind, x, y in samples:
            w.writerow(["{:.1f}".format(t), kind, "{:.1f}".format(x), "{:.1f}".format(y)])
    canvas = _field_canvas(cv2, np, size)
    colors = {"pollen": (0, 220, 255), "red_nectar": (40, 40, 255), "blue_nectar": (255, 120, 40)}
    overlay = canvas.copy()
    for kind, col in colors.items():
        for _t, k, x, y in samples:
            if k == kind:
                cv2.circle(overlay, _to_px(x, y, size), 4, col, -1, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
    # which side runs dry first: share of POLLEN on each half, early vs late
    lines = []
    times = [t for t, k, _x, _y in samples if k == "pollen"]
    if times:
        mid = (min(times) + max(times)) / 2.0
        for label, sel in (("first half of match", lambda t: t < mid), ("second half", lambda t: t >= mid)):
            pts = [x for t, k, x, _y in samples if k == "pollen" and sel(t)]
            if pts:
                left = sum(1 for x in pts if x < 0) / len(pts)
                lines.append("POLLEN {}: {:.0f}% left / {:.0f}% right".format(label, 100 * left, 100 * (1 - left)))
    lines.insert(0, "Loose pieces: POLLEN yellow, red NECTAR red, blue NECTAR blue")
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(png_path, canvas)


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_all(cv2, np, output_dir, csv_path, tip_rows, shot_rows, resource_samples):
    with open(csv_path, newline="") as handle:
        pose_rows = list(csv.DictReader(handle))
    alliances = alliance_summary(tip_rows)
    robots = robot_summary(pose_rows, shot_rows)
    write_rows(os.path.join(output_dir, "alliance_summary.csv"), alliances)
    write_rows(os.path.join(output_dir, "robot_summary.csv"), robots)
    write_shot_heatmap(cv2, np, os.path.join(output_dir, "shot_heatmap.png"), shot_rows)
    write_resource_density(cv2, np, os.path.join(output_dir, "resource_density.png"),
                           os.path.join(output_dir, "resource_density.csv"), resource_samples)
    return alliances, robots
