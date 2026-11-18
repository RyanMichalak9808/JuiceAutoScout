import math
from typing import Dict, List, Optional, Tuple

from autoscout import season
from autoscout.models import BallTrack, ShotEvent


class ShotDetector:
    # Pixel constants below are tuned for REF_FRAME_WIDTH-wide video and are
    # scaled linearly (areas quadratically) to the actual frame width.
    REF_FRAME_WIDTH = 640.0
    # BIOBUZZ POLLEN (2.8 in) is much smaller than DECODE balls.
    BALL_MIN_CONTOUR_AREA_PX = 3.0
    BALL_MAX_CONTOUR_AREA_PX = 200.0
    BALL_ASSOC_DIST_PX = 34.0
    BALL_ASSOC_DIST_FAST_PX = 124.0
    BALL_MAX_MISSING = 12
    SHOT_MAX_START_DIST_PX = 84.0
    SHOT_MIN_DISP_PX = 8.0
    SHOT_MIN_SPEED_PX = 2.0
    SHOT_MIN_UPWARD_PX = 3.0
    SHOT_MIN_LIFE_FRAMES = 2
    SHOT_MIN_NEGATIVE_STEP_RATIO = 0.35
    SHOT_MIN_AWAY_FROM_SHOOTER_PX = 2.0
    SHOT_DEDUP_TIME_S = 0.35
    GOAL_APPROACH_SIDE_PAD_FRAC = 0.30
    GOAL_APPROACH_DOWN_FRAC = 0.95
    # Calibrated HIVE boxes already include the space around the cells, so
    # the "approaching" zone around them is much smaller.
    BOX_APPROACH_SIDE_PAD_FRAC = 0.15
    BOX_APPROACH_DOWN_FRAC = 0.25
    GOAL_CONFIRM_MIN_FRAMES = 2
    GOAL_ENTRY_MAX_MISSING = 5
    # Clear HIVE cells: a made ball stays visible and comes to rest inside the
    # target. Resolve as made once it has moved less than GOAL_SETTLE_PX per
    # frame for GOAL_SETTLE_FRAMES frames inside the target polygon.
    GOAL_SETTLE_FRAMES = 4
    GOAL_SETTLE_PX = 2.0
    # Color-based HIVE fallback (only used without hive_boxes_px):
    GOAL_TOP_BAND_FRAC = 0.55          # search the top part of the frame
    GOAL_MAX_TOP_FRAC = 0.35           # target top edge must be above this
    GOAL_SEARCH_HALF_WIDTH_FRAC = 0.30 # search this far either side of field center
    GOAL_JOIN_KERNEL_PX = 9            # joins the thin HIVE cell outlines
    GOAL_OPENING_SHRINK_X = 0.90
    GOAL_OPENING_SHRINK_Y = 0.90
    GOAL_BLUE_HSV_LO = (95, 80, 45)
    GOAL_BLUE_HSV_HI = (135, 255, 255)
    GOAL_RED1_HSV_LO = (0, 90, 45)
    GOAL_RED1_HSV_HI = (12, 255, 255)
    GOAL_RED2_HSV_LO = (170, 90, 45)
    GOAL_RED2_HSV_HI = (179, 255, 255)
    GOAL_MIN_AREA_PX = 900.0

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self._next_track_id = 1
        self._tracks: Dict[int, BallTrack] = {}
        self._goal_openings: Dict[str, object] = {}
        self._goal_approaches: Dict[str, object] = {}
        self._last_event_time_by_robot: Dict[int, float] = {}
        self._ready = False
        self._lin = 1.0   # linear pixel scale vs REF_FRAME_WIDTH

    def _px(self, value: float) -> float:
        return value * self._lin

    def _area(self, value: float) -> float:
        return value * self._lin * self._lin

    def setup(self, tracker) -> None:
        if tracker._bg is None or tracker._field_mask is None:
            return
        self._lin = tracker._bg.shape[1] / self.REF_FRAME_WIDTH
        if tracker.hive_boxes:
            openings = {color: poly.copy() for color, poly in tracker.hive_boxes.items()}
            print("[INFO] Shot detector using calibrated HIVE boxes.")
        else:
            openings = self._detect_goal_openings(tracker._bg, tracker)
            if openings:
                print("[WARN] No hive_boxes_px in calibration; HIVES found by color. "
                      "Draw HIVE boxes in tools/calibrate.py for reliable results.")
        if not openings:
            print("[WARN] Shot detector could not determine HIVE targets. "
                  "Draw HIVE boxes with tools/calibrate.py (key H).")
            return
        self._goal_openings = openings
        if tracker.hive_boxes:
            side_pad, down = self.BOX_APPROACH_SIDE_PAD_FRAC, self.BOX_APPROACH_DOWN_FRAC
        else:
            side_pad, down = self.GOAL_APPROACH_SIDE_PAD_FRAC, self.GOAL_APPROACH_DOWN_FRAC
        self._goal_approaches = {
            color: self._build_goal_approach(poly, side_pad, down)
            for color, poly in openings.items()
        }
        self._ready = True
        print("[INFO] Shot detector goal openings ready: {}".format(
            ", ".join(sorted(openings.keys()))))

    def update(self, frame_num: int, match_time_s: float, poses, tracker) -> List[ShotEvent]:
        if not self._ready or tracker._last_ball_mask is None or tracker._H_inv is None:
            return []

        detections = self._extract_ball_detections(tracker._last_ball_mask)
        robot_refs = self._robot_refs(poses, tracker)
        self._associate_tracks(detections, robot_refs, frame_num)

        events = []
        for track in list(self._tracks.values()):
            if not track.samples or track.resolved:
                continue
            self._update_goal_state(track)
            self._maybe_mark_launched(track, robot_refs)
            if track.launched and self._settled_in_goal(track):
                event = self._resolve_track(track, frame_num, match_time_s)
                if event is not None and not self._is_duplicate_event(event):
                    events.append(event)
                del self._tracks[track.track_id]
                continue
            if track.missing_frames > self.BALL_MAX_MISSING:
                event = self._resolve_track(track, frame_num, match_time_s)
                if event is not None and not self._is_duplicate_event(event):
                    events.append(event)
                del self._tracks[track.track_id]
        return events

    def draw_debug(self, dbg) -> None:
        if not self._ready:
            return
        cv2 = self.cv2
        for color, poly in self._goal_approaches.items():
            cv_color = (255, 180, 60) if color == "blue" else (80, 180, 255)
            cv2.polylines(dbg, [poly], True, cv_color, 1, cv2.LINE_AA)
        for color, poly in self._goal_openings.items():
            cv_color = (255, 80, 0) if color == "blue" else (0, 80, 255)
            cv2.polylines(dbg, [poly], True, cv_color, 2, cv2.LINE_AA)
        for track in self._tracks.values():
            pts = [(int(round(x)), int(round(y))) for _f, x, y in track.samples[-8:]]
            if len(pts) >= 2:
                cv2.polylines(
                    dbg,
                    [self.np.array(pts, dtype=self.np.int32).reshape(-1, 1, 2)],
                    False,
                    (0, 255, 255) if track.launched else (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )
            if pts:
                px, py = pts[-1]
                label = "S{}".format(track.shooter_id) if track.launched and track.shooter_id is not None else "b"
                if track.entered_goal:
                    label += ":goal"
                cv2.putText(dbg, label, (px + 4, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _extract_ball_detections(self, ball_mask):
        cv2 = self.cv2
        cnts, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            if len(c) < 3:
                # contourArea is 0 for 1-2 pixel blobs; use the pixel count.
                area = float(len(c))
            if area < self._area(self.BALL_MIN_CONTOUR_AREA_PX) or area > self._area(self.BALL_MAX_CONTOUR_AREA_PX):
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                pts = c.reshape(-1, 2)
                cx = float(pts[:, 0].mean())
                cy = float(pts[:, 1].mean())
                detections.append((cx, cy, area))
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            detections.append((cx, cy, area))
        return detections

    def _robot_refs(self, poses, tracker):
        refs = []
        for rid, pose in enumerate(poses):
            if not pose.visible:
                continue
            ix, iy = tracker.field_to_image(pose.x_in, pose.y_in)
            refs.append({
                "id": rid,
                "img": (ix, iy),
                "center": (float(pose.x_center_in), float(pose.y_center_in)),
            })
        return refs

    def _associate_tracks(self, detections, robot_refs, frame_num: int) -> None:
        unmatched_tracks = set(self._tracks.keys())
        unmatched_dets = set(range(len(detections)))
        pairs = []
        for tid, track in self._tracks.items():
            pred_x, pred_y = self._predict_track_point(track)
            max_dist = self._px(self.BALL_ASSOC_DIST_FAST_PX if track.launched else self.BALL_ASSOC_DIST_PX)
            for di, (cx, cy, _area) in enumerate(detections):
                d = math.hypot(cx - pred_x, cy - pred_y)
                if d <= max_dist:
                    pairs.append((d, tid, di))
        pairs.sort()
        for _d, tid, di in pairs:
            if tid not in unmatched_tracks or di not in unmatched_dets:
                continue
            cx, cy, _area = detections[di]
            track = self._tracks[tid]
            track.samples.append((frame_num, cx, cy))
            track.missing_frames = 0
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(di)

        for tid in unmatched_tracks:
            self._tracks[tid].missing_frames += 1

        for di in unmatched_dets:
            cx, cy, _area = detections[di]
            nearest_id = None
            nearest_dist = None
            nearest_img = None
            nearest_center = None
            for ref in robot_refs:
                dist = math.hypot(cx - ref["img"][0], cy - ref["img"][1])
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = ref["id"]
                    nearest_img = ref["img"]
                    nearest_center = ref["center"]
            self._tracks[self._next_track_id] = BallTrack(
                track_id=self._next_track_id,
                samples=[(frame_num, cx, cy)],
                shooter_id=nearest_id,
                shooter_img=nearest_img,
                shooter_dist_px=nearest_dist,
                shooter_pos_center_in=nearest_center,
            )
            self._next_track_id += 1

    def _predict_track_point(self, track: BallTrack) -> Tuple[float, float]:
        if len(track.samples) < 2:
            _f, x, y = track.samples[-1]
            return x, y
        _f1, x1, y1 = track.samples[-1]
        _f0, x0, y0 = track.samples[-2]
        return x1 + (x1 - x0), y1 + (y1 - y0)

    def _update_goal_state(self, track: BallTrack) -> None:
        frame_idx, x, y = track.samples[-1]
        for color, poly in self._goal_approaches.items():
            if self.cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                track.approached_goal = True
                if not track.goal_color:
                    track.goal_color = color
        for color, poly in self._goal_openings.items():
            if self.cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                track.entered_goal = True
                track.entered_goal_frames += 1
                if track.first_goal_entry_frame is None:
                    track.first_goal_entry_frame = frame_idx
                track.last_goal_entry_point = (float(x), float(y))
                track.goal_color = color
                break

    def _maybe_mark_launched(self, track: BallTrack, robot_refs) -> None:
        if track.launched or len(track.samples) < self.SHOT_MIN_LIFE_FRAMES:
            return
        first = track.first_point
        last = track.last_point
        if first is None or last is None:
            return
        shooter_ref, shooter_score = self._select_shooter(track, robot_refs)
        if shooter_ref is None or shooter_score > self._px(self.SHOT_MAX_START_DIST_PX):
            return
        f0, x0, y0 = first
        f1, x1, y1 = last
        dt = max(f1 - f0, 1)
        disp = math.hypot(x1 - x0, y1 - y0)
        avg_speed = disp / dt
        if disp < self._px(self.SHOT_MIN_DISP_PX) or avg_speed < self._px(self.SHOT_MIN_SPEED_PX):
            return
        if (y1 - y0) > -self._px(self.SHOT_MIN_UPWARD_PX):
            return
        if not self._has_consistent_upward_motion(track):
            return
        if not self._moving_away_from_shooter(track):
            return
        track.shooter_id = shooter_ref["id"]
        track.shooter_img = shooter_ref["img"]
        track.shooter_dist_px = shooter_score
        track.shooter_pos_center_in = shooter_ref["center"]
        track.launched = True

    def _resolve_track(self, track: BallTrack, frame_num: int, match_time_s: float) -> Optional[ShotEvent]:
        if not track.launched or track.shooter_id is None or track.shooter_pos_center_in is None:
            return None
        if self._confirmed_goal_make(track):
            result = "made"
        elif track.approached_goal or self._track_reached_top(track):
            result = "missed"
        else:
            return None
        sx, sy = track.shooter_pos_center_in
        track.resolved = True
        return ShotEvent(
            shooter_id=track.shooter_id,
            result=result,
            shot_x_in=float(sx),
            shot_y_in=float(sy),
            frame_num=frame_num,
            timestamp_s=match_time_s,
            goal_color=track.goal_color,
        )

    def _has_consistent_upward_motion(self, track: BallTrack) -> bool:
        if len(track.samples) < 3:
            return False
        negative_steps = 0
        total_steps = 0
        for (_f0, _x0, y0), (_f1, _x1, y1) in zip(track.samples[:-1], track.samples[1:]):
            total_steps += 1
            if y1 < y0:
                negative_steps += 1
        return total_steps > 0 and (negative_steps / total_steps) >= self.SHOT_MIN_NEGATIVE_STEP_RATIO

    def _select_shooter(self, track: BallTrack, robot_refs):
        if not robot_refs:
            return None, float("inf")
        first = track.first_point
        if first is None:
            return None, float("inf")
        _f0, x0, y0 = first
        origin_x, origin_y = x0, y0
        if len(track.samples) >= 2:
            deltas = []
            for (_fa, xa, ya), (_fb, xb, yb) in zip(track.samples[:-1], track.samples[1:]):
                deltas.append((xb - xa, yb - ya))
                if len(deltas) >= 3:
                    break
            if deltas:
                avg_dx = sum(dx for dx, _dy in deltas) / len(deltas)
                avg_dy = sum(dy for _dx, dy in deltas) / len(deltas)
                origin_x = x0 - 1.35 * avg_dx
                origin_y = y0 - 1.35 * avg_dy

        best_ref = None
        best_score = float("inf")
        for ref in robot_refs:
            rx, ry = ref["img"]
            dist_origin = math.hypot(origin_x - rx, origin_y - ry)
            dist_first = math.hypot(x0 - rx, y0 - ry)
            above_penalty = max(0.0, (y0 - ry) - self._px(6.0)) * 0.75
            score = 0.7 * dist_origin + 0.3 * dist_first + above_penalty
            if score < best_score:
                best_score = score
                best_ref = ref
        return best_ref, best_score

    def _moving_away_from_shooter(self, track: BallTrack) -> bool:
        if track.shooter_img is None or len(track.samples) < 2:
            return False
        first = track.first_point
        last = track.last_point
        if first is None or last is None:
            return False
        sx, sy = track.shooter_img
        _f0, x0, y0 = first
        _f1, x1, y1 = last
        start_dist = math.hypot(x0 - sx, y0 - sy)
        last_dist = math.hypot(x1 - sx, y1 - sy)
        return last_dist >= start_dist + self._px(self.SHOT_MIN_AWAY_FROM_SHOOTER_PX)

    def _confirmed_goal_make(self, track: BallTrack) -> bool:
        if not track.entered_goal:
            return False
        if track.entered_goal_frames >= self.GOAL_CONFIRM_MIN_FRAMES:
            return True
        if track.first_goal_entry_frame is None:
            return False
        if track.missing_frames <= self.GOAL_ENTRY_MAX_MISSING and self._track_finished_inside_goal(track):
            return True
        return False

    def _track_finished_inside_goal(self, track: BallTrack) -> bool:
        if not track.goal_color or track.last_goal_entry_point is None:
            return False
        poly = self._goal_openings.get(track.goal_color)
        if poly is None:
            return False
        x, y = track.last_goal_entry_point
        return self.cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def _is_duplicate_event(self, event: ShotEvent) -> bool:
        last_time = self._last_event_time_by_robot.get(event.shooter_id)
        if last_time is not None and (event.timestamp_s - last_time) < self.SHOT_DEDUP_TIME_S:
            return True
        self._last_event_time_by_robot[event.shooter_id] = event.timestamp_s
        return False

    def _track_reached_top(self, track: BallTrack) -> bool:
        if not track.samples:
            return False
        top_y = min(y for _f, _x, y in track.samples)
        goal_top = min(
            min(poly[:, 0, 1]) for poly in self._goal_approaches.values()
        ) if self._goal_approaches else 0.0
        goal_bottom = max(
            max(poly[:, 0, 1]) for poly in self._goal_approaches.values()
        ) if self._goal_approaches else 0.0
        return top_y <= goal_bottom + max(self._px(14.0), 0.2 * max(goal_bottom - goal_top, 1.0))

    def _settled_in_goal(self, track: BallTrack) -> bool:
        n = self.GOAL_SETTLE_FRAMES
        if not track.entered_goal or len(track.samples) < n + 1 or track.missing_frames > 0:
            return False
        recent = track.samples[-(n + 1):]
        poly = self._goal_openings.get(track.goal_color)
        if poly is None:
            return False
        limit = self._px(self.GOAL_SETTLE_PX)
        for (_fa, xa, ya), (_fb, xb, yb) in zip(recent[:-1], recent[1:]):
            if math.hypot(xb - xa, yb - ya) > limit:
                return False
            if self.cv2.pointPolygonTest(poly, (float(xb), float(yb)), False) < 0:
                return False
        return True

    def _detect_goal_openings(self, background, tracker):
        """Fallback: find each alliance's scoring target by color in the
        median background. Targets sit near the field center horizontally
        (season.HIVE_SIDES says which side of center each alliance is on)
        and in the upper part of the frame."""
        cv2, np = self.cv2, self.np
        hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        h, w = background.shape[:2]
        top_band = np.zeros((h, w), dtype=np.uint8)
        top_band[:max(1, int(round(h * self.GOAL_TOP_BAND_FRAC))), :] = 255

        center_u, _center_v = tracker.field_to_image(72.0, 72.0) if tracker._H_inv is not None else (w / 2.0, 0.0)
        split = int(round(min(max(center_u, 0.0), w - 1.0)))
        half = int(round(w * self.GOAL_SEARCH_HALF_WIDTH_FRAC))

        masks = {
            "blue": cv2.inRange(
                hsv,
                np.array(self.GOAL_BLUE_HSV_LO, dtype=np.uint8),
                np.array(self.GOAL_BLUE_HSV_HI, dtype=np.uint8),
            ),
            "red": cv2.bitwise_or(
                cv2.inRange(
                    hsv,
                    np.array(self.GOAL_RED1_HSV_LO, dtype=np.uint8),
                    np.array(self.GOAL_RED1_HSV_HI, dtype=np.uint8),
                ),
                cv2.inRange(
                    hsv,
                    np.array(self.GOAL_RED2_HSV_LO, dtype=np.uint8),
                    np.array(self.GOAL_RED2_HSV_HI, dtype=np.uint8),
                ),
            ),
        }

        openings = {}
        for color, mask in masks.items():
            mask = cv2.bitwise_and(mask, top_band)
            side = season.HIVE_SIDES.get(color, "left")
            keep = np.zeros_like(mask)
            if side == "left":
                keep[:, max(0, split - half):split] = 255
            else:
                keep[:, split:min(w, split + half)] = 255
            mask = cv2.bitwise_and(mask, keep)
            small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            join_size = max(3, int(round(self._px(self.GOAL_JOIN_KERNEL_PX))) | 1)
            join = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (join_size, join_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, join)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            chosen = None
            for c in cnts:
                if float(cv2.contourArea(c)) < self._area(self.GOAL_MIN_AREA_PX):
                    continue
                _x, y, _ww, _hh = cv2.boundingRect(c)
                if y > int(self.GOAL_MAX_TOP_FRAC * h):
                    continue
                chosen = c
                break
            if chosen is None:
                continue
            openings[color] = self._shrink_polygon(
                cv2.convexHull(chosen),
                shrink_x=self.GOAL_OPENING_SHRINK_X,
                shrink_y=self.GOAL_OPENING_SHRINK_Y,
            )
        return openings

    def _build_goal_approach(self, poly, side_pad_frac, down_frac):
        pts = poly.reshape(-1, 2).astype(self.np.float32)
        x_min = float(self.np.min(pts[:, 0]))
        x_max = float(self.np.max(pts[:, 0]))
        y_min = float(self.np.min(pts[:, 1]))
        y_max = float(self.np.max(pts[:, 1]))
        w = max(x_max - x_min, 1.0)
        h = max(y_max - y_min, 1.0)
        pad_x = w * side_pad_frac
        down = h * down_frac
        rect = self.np.array([
            [x_min - pad_x, y_min],
            [x_max + pad_x, y_min],
            [x_max + pad_x, y_max + down],
            [x_min - pad_x, y_max + down],
        ], dtype=self.np.int32)
        return rect.reshape(-1, 1, 2)

    def _shrink_polygon(self, poly, shrink_x: float, shrink_y: float):
        pts = poly.reshape(-1, 2).astype(self.np.float32)
        cx = float(self.np.mean(pts[:, 0]))
        cy = float(self.np.mean(pts[:, 1]))
        scaled = pts.copy()
        scaled[:, 0] = cx + (scaled[:, 0] - cx) * shrink_x
        scaled[:, 1] = cy + (scaled[:, 1] - cy) * shrink_y
        return self.np.round(scaled).astype(self.np.int32).reshape(-1, 1, 2)
