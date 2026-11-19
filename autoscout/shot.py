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
    BALL_MAX_ELONGATION = 3.5          # bbox long side / short side (motion blur allowed)
    # Ball tracking runs on every source frame (pixels at 640 px width).
    GRAVITY_PX_PER_FRAME2 = 0.6        # ball acceleration downward in the image
    GATE_X_PX = 7.0                    # association window around the prediction
    GATE_Y_PX = 9.0
    GATE_NEW_X_PX = 7.0                # for a track with one sample, per frame
    GATE_NEW_UP_PX = 24.0
    GATE_NEW_DOWN_PX = 6.0
    REFS_EVERY_FRAMES = 1
    TRAIL_KEEP_FRAMES = 20             # keep finished shot trails in the debug video
    # A shot: the ball rises at least SHOT_MIN_RISE_PX, mostly in upward
    # steps, and moves sideways at most SHOT_MAX_LATERAL_PER_RISE x its rise.
    SHOT_MIN_SAMPLES = 3
    SHOT_MIN_RISE_PX = 14.0
    SHOT_MIN_UP_STEP_RATIO = 0.6
    SHOT_MAX_LATERAL_PER_RISE = 1.5
    SHOT_MAX_START_ROBOTS = 1.6        # shooter within this many robot sizes of the launch point
    # The CELL zone of a HIVE box: its upper part. A ball reaching it counts
    # as entering the HIVE.
    CELL_ZONE_FRAC = 0.8
    BALL_ASSOC_DIST_PX = 34.0
    # Max jump (px at REF_FRAME_WIDTH) from the predicted position for a ball
    # already marked launched. 124 chained unrelated POLLEN into one "shot"
    # on BIOBUZZ footage.
    BALL_ASSOC_DIST_FAST_PX = 70.0
    BALL_MAX_MISSING = 8               # source frames a ball may be unseen
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
    GOAL_SETTLE_FRAMES = 8
    GOAL_SETTLE_PX = 1.5
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
        self._refs = []
        self._refs_frame = -10**9
        self._recent_launched: List[BallTrack] = []
        self._seen_event_keys = set()
        self._cell_zone_bottom: Dict[str, float] = {}
        self._last_frame = 0
        self._fps = 30.0

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
        for color, poly in openings.items():
            ys = poly.reshape(-1, 2)[:, 1]
            self._cell_zone_bottom[color] = float(ys.min() + self.CELL_ZONE_FRAC * (ys.max() - ys.min()))
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

    def update(self, frame_num: int, match_time_s: float, poses, tracker,
               hive_moving=None) -> List[ShotEvent]:
        """Call for EVERY source frame (tracker.ball_mask_only() on frames the
        robot tracker skips) so fast vertical shots are followed frame by
        frame. hive_moving: optional callable(color) -> bool, True while that
        HIVE is tipping."""
        if not self._ready or tracker._last_ball_mask is None or tracker._H_inv is None:
            return []
        self._hive_moving = hive_moving
        if poses is not None and (frame_num - self._refs_frame >= self.REFS_EVERY_FRAMES or not self._refs):
            self._refs = self._robot_refs(poses, tracker)
            self._refs_frame = frame_num
        detections = self._extract_ball_detections(tracker._last_ball_mask)
        self._associate_tracks(detections, frame_num)

        events = []
        for track in list(self._tracks.values()):
            if track.resolved:
                continue
            if track.missing_frames == 0:
                self._update_goal_state(track)
                self._maybe_mark_launched(track, tracker)
            ended = track.missing_frames > self.BALL_MAX_MISSING
            settled = track.launched and self._settled_in_goal(track)
            if ended or settled:
                event = self._resolve_track(track, frame_num, match_time_s)
                if event is not None and not self._is_duplicate_event(event):
                    events.append(event)
                if track.launched:
                    self._recent_launched.append(track)
                    self._recent_launched = self._recent_launched[-12:]
                del self._tracks[track.track_id]
        return events

    def draw_debug(self, dbg) -> None:
        if not self._ready:
            return
        cv2, np = self.cv2, self.np
        for color, poly in self._goal_openings.items():
            cv_color = (255, 80, 0) if color == "blue" else (0, 80, 255)
            cv2.polylines(dbg, [poly], True, cv_color, 2, cv2.LINE_AA)
            rim = int(round(self._cell_zone_bottom[color]))
            pts = poly.reshape(-1, 2)
            cv2.line(dbg, (int(pts[:, 0].min()), rim), (int(pts[:, 0].max()), rim), cv_color, 1, cv2.LINE_AA)
        tracks = list(self._tracks.values()) + [t for t in self._recent_launched if t.samples
                                                 and self._last_frame - t.samples[-1][0] <= self.TRAIL_KEEP_FRAMES]
        for track in tracks:
            pts = [(int(round(x)), int(round(y))) for _f, x, y in track.samples[-30:]]
            if len(pts) < 2 and not track.launched:
                continue
            if track.launched:
                col = (0, 255, 255)                         # yellow: shot
                if track.result == "made":
                    col = (0, 255, 0)
                elif track.result == "missed":
                    col = (0, 0, 255)
            else:
                col = (150, 150, 150)                       # gray: ball not (yet) a shot
            if len(pts) >= 2:
                cv2.polylines(dbg, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], False, col,
                              2 if track.launched else 1, cv2.LINE_AA)
            px, py = pts[-1]
            if track.launched:
                label = "S{}".format(track.shooter_id) if track.shooter_id is not None and track.shooter_id >= 0 else "S?"
                if track.result:
                    label += " " + track.result
                cv2.putText(dbg, label, (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    def _extract_ball_detections(self, ball_mask):
        cv2 = self.cv2
        cnts, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        amin = self._area(self.BALL_MIN_CONTOUR_AREA_PX)
        amax = self._area(self.BALL_MAX_CONTOUR_AREA_PX)
        for c in cnts:
            area = max(float(cv2.contourArea(c)), float(len(c)) if len(c) < 3 else 0.0)
            if area < amin or area > amax:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # a ball (possibly motion-blurred along its flight) is compact;
            # long thin shapes are edges of people, rails, robot parts
            if max(bw, bh) > self.BALL_MAX_ELONGATION * min(bw, bh) + self._px(3):
                continue
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            else:
                cx, cy = x + bw / 2.0, y + bh / 2.0
            detections.append((float(cx), float(cy), area))
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
                "size": tracker._robot_px_at(ix, iy),
                "center": (float(pose.x_center_in), float(pose.y_center_in)),
            })
        return refs

    def _predict(self, track: BallTrack, frame_num: int):
        f1, x1, y1 = track.samples[-1]
        dt = max(1, frame_num - f1)
        if len(track.samples) < 2:
            return x1, y1, dt, None
        f0, x0, y0 = track.samples[-2]
        step = max(1, f1 - f0)
        vx, vy = (x1 - x0) / step, (y1 - y0) / step
        g = self._px(self.GRAVITY_PX_PER_FRAME2)
        return x1 + vx * dt, y1 + vy * dt + 0.5 * g * dt * dt, dt, (vx, vy)

    def _associate_tracks(self, detections, frame_num: int) -> None:
        self._last_frame = frame_num
        pairs = []
        for tid, track in self._tracks.items():
            px, py, dt, vel = self._predict(track, frame_num)
            if vel is None:
                # one sample so far: the ball may go anywhere upward fast, or
                # drift slowly; allow a tall window biased upward
                gx = self._px(self.GATE_NEW_X_PX) * dt + self._px(4)
                up = self._px(self.GATE_NEW_UP_PX) * dt + self._px(4)
                down = self._px(self.GATE_NEW_DOWN_PX) * dt + self._px(4)
                for di, (cx, cy, _a) in enumerate(detections):
                    dx, dy = cx - px, cy - py
                    if abs(dx) <= gx and -up <= dy <= down:
                        cost = (dx / gx) ** 2 + (dy / (up if dy < 0 else down)) ** 2
                        pairs.append((cost, tid, di))
                continue
            vx, vy = vel
            gx = self._px(self.GATE_X_PX) + 0.35 * abs(vx) * dt
            gy = self._px(self.GATE_Y_PX) + 0.5 * abs(vy) * dt + self._px(self.GRAVITY_PX_PER_FRAME2) * dt * dt
            for di, (cx, cy, _a) in enumerate(detections):
                dx, dy = cx - px, cy - py
                if abs(dx) <= gx and abs(dy) <= gy:
                    pairs.append(((dx / gx) ** 2 + (dy / gy) ** 2, tid, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _cost, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            cx, cy, _a = detections[di]
            track = self._tracks[tid]
            track.samples.append((frame_num, cx, cy))
            track.missing_frames = 0
            used_t.add(tid)
            used_d.add(di)
        for tid, track in self._tracks.items():
            if tid not in used_t:
                track.missing_frames = frame_num - track.samples[-1][0]
        for di, (cx, cy, _a) in enumerate(detections):
            if di in used_d:
                continue
            self._tracks[self._next_track_id] = BallTrack(
                track_id=self._next_track_id, samples=[(frame_num, cx, cy)])
            self._next_track_id += 1

    def _in_cell_zone(self, color: str, x: float, y: float) -> bool:
        poly = self._goal_openings[color]
        return (self.cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                and y <= self._cell_zone_bottom[color])

    def _update_goal_state(self, track: BallTrack) -> None:
        frame_idx, x, y = track.samples[-1]
        for color, poly in self._goal_approaches.items():
            if self.cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                track.approached_goal = True
                if not track.goal_color:
                    track.goal_color = color
        for color in self._goal_openings:
            if self._in_cell_zone(color, x, y):
                track.entered_goal = True
                track.entered_goal_frames += 1
                if track.first_goal_entry_frame is None:
                    track.first_goal_entry_frame = frame_idx
                track.last_goal_entry_point = (float(x), float(y))
                track.goal_color = color
                break

    def _starts_in_cells(self, track: BallTrack) -> Optional[str]:
        _f, x, y = track.samples[0]
        for color in self._goal_openings:
            if self._in_cell_zone(color, x, y):
                return color
        return None

    def _maybe_mark_launched(self, track: BallTrack, tracker) -> None:
        """A shot = a ball that rises clearly: net upward travel of at least
        SHOT_MIN_RISE_PX, mostly upward steps, more vertical than sideways.
        Shots this season are steep, so vertical motion is what counts."""
        if track.launched or len(track.samples) < self.SHOT_MIN_SAMPLES:
            return
        if self._starts_in_cells(track) is not None:
            # balls first seen inside a CELL (resting, bouncing, or moving
            # with a tipping HIVE) are not launches
            return
        ys = [y for _f, _x, y in track.samples]
        xs = [x for _f, x, _y in track.samples]
        k_top = int(min(range(len(ys)), key=lambda k: ys[k]))
        rise = ys[0] - ys[k_top]
        if rise < self._px(self.SHOT_MIN_RISE_PX):
            return
        lateral = abs(xs[k_top] - xs[0])
        if lateral > self.SHOT_MAX_LATERAL_PER_RISE * rise:
            return
        steps = list(zip(ys[:k_top], ys[1:k_top + 1]))
        if steps:
            up = sum(1 for a, b in steps if b < a)
            if up / len(steps) < self.SHOT_MIN_UP_STEP_RATIO:
                return
        track.launched = True
        shooter, dist = self._select_shooter(track)
        if shooter is not None:
            track.shooter_id = shooter["id"]
            track.shooter_img = shooter["img"]
            track.shooter_dist_px = dist
            track.shooter_pos_center_in = shooter["center"]
        else:
            track.shooter_id = -1

    def _select_shooter(self, track: BallTrack):
        """Robot the ball left from: the visible robot nearest to where the
        flight started (extrapolated a little backward), preferring robots
        below the ball. Within SHOT_MAX_START_ROBOTS robot sizes, else none."""
        if not self._refs:
            return None, None
        _f0, x0, y0 = track.samples[0]
        if len(track.samples) >= 2:
            _f1, x1, y1 = track.samples[1]
            x0, y0 = x0 - 0.7 * (x1 - x0), y0 - 0.7 * (y1 - y0)
        best, best_score = None, None
        for ref in self._refs:
            rx, ry = ref["img"]
            size = max(ref["size"], 1.0)
            d = math.hypot(x0 - rx, (y0 - ry) * 0.7) / size   # robots are tall: forgive vertical offset
            if y0 > ry + 0.6 * size:
                d += 1.0                                       # ball starting well below the robot
            if best_score is None or d < best_score:
                best, best_score = ref, d
        if best is None or best_score > self.SHOT_MAX_START_ROBOTS:
            return None, None
        return best, best_score

    def _settled_in_goal(self, track: BallTrack) -> bool:
        n = self.GOAL_SETTLE_FRAMES
        if not track.entered_goal or len(track.samples) < n + 1 or track.missing_frames > 0:
            return False
        recent = track.samples[-(n + 1):]
        limit = self._px(self.GOAL_SETTLE_PX)
        for (fa, xa, ya), (fb, xb, yb) in zip(recent[:-1], recent[1:]):
            if math.hypot(xb - xa, yb - ya) > limit * max(1, fb - fa):
                return False
            if not self._in_cell_zone(track.goal_color, xb, yb):
                return False
        return True

    def _resolve_track(self, track: BallTrack, frame_num: int, match_time_s: float) -> Optional[ShotEvent]:
        if not track.launched:
            return None
        result = None
        if track.entered_goal and track.goal_color:
            # made unless the ball was then seen falling out below the CELLS
            after = [(f, x, y) for f, x, y in track.samples if track.first_goal_entry_frame is not None
                     and f >= track.first_goal_entry_frame]
            fell_out = False
            poly = self._goal_openings[track.goal_color]
            box_bottom = float(poly.reshape(-1, 2)[:, 1].max())
            for (fa, xa, ya), (fb, xb, yb) in zip(after[:-1], after[1:]):
                if yb > box_bottom + self._px(4) and yb > ya:
                    fell_out = True
                    break
            result = "missed" if fell_out else "made"
        elif track.approached_goal or self._track_reached_top(track):
            result = "missed"
        if result is None:
            return None
        track.resolved = True
        track.result = result
        sx, sy = track.shooter_pos_center_in if track.shooter_pos_center_in is not None else (float("nan"), float("nan"))
        return ShotEvent(
            shooter_id=track.shooter_id if track.shooter_id is not None else -1,
            result=result,
            shot_x_in=float(sx),
            shot_y_in=float(sy),
            frame_num=track.samples[0][0],
            timestamp_s=match_time_s - (frame_num - track.samples[0][0]) / max(self._fps, 1.0),
            goal_color=track.goal_color,
        )

    def _is_duplicate_event(self, event: ShotEvent) -> bool:
        # Robots fire POLLEN in quick bursts, so only drop events that start
        # in the same frame for the same robot (one ball seen as two tracks).
        key = (event.shooter_id, event.frame_num, event.goal_color, event.result)
        if key in self._seen_event_keys:
            return True
        self._seen_event_keys.add(key)
        return False

    def _track_reached_top(self, track: BallTrack) -> bool:
        if not track.samples or not self._goal_openings:
            return False
        top_y = min(y for _f, _x, y in track.samples)
        rim = max(self._cell_zone_bottom.values())
        return top_y <= rim

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
