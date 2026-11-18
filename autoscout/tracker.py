import itertools
import math
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Tuple

from autoscout import season
from autoscout.models import MergeGroup, RobotPose
from autoscout.runtime import _make_bar


class RobotTracker:
    N_BG_SAMPLES  = 80
    FG_THRESH     = 30       # Raised from 22: reduces noise from crowd/alliance members
    BLOB_MIN      = 350      # Raised from 260: skip tiny noise blobs
    MIN_RADIUS_PX = 6        # Lowered from 8: far-wall robots appear smaller
    KERNEL_PX     = 7        # Lowered from 9: better for 640x360 video resolution
    MAX_COAST     = 60
    MAX_DIST_IN   = 30.0
    ROBOT_SIZE_IN = 18.0
    MAX_SPEED_IN  = 140.0    # Raised from 120: allow slightly faster apparent motion
    MAX_REACQ_IN  = 144.0
    REACQ_PX_PAD  = 2.0      # Raised from 1.6: wider reacquisition window
    VISIBLE_COAST = 8
    MERGE_HOLD    = 16
    MERGE_DEBUG_HOLD = 0
    EXPECT_2WAY_AREA = 1.6
    EXPECT_3WAY_AREA = 2.6
    EXPECT_4WAY_AREA = 3.5
    RELAXED_PEAK_RATIO_MULTI = 0.30
    RELAXED_PEAK_RATIO_PAIR  = 0.35
    RELAXED_SEP_MULTI = 3
    RELAXED_SEP_PAIR  = 5
    MERGE_PRIOR_PAD_PX = 0.75
    MERGE_PRIOR_MIN_SEP = 0.35
    BLOB_MIN_FIELD_AREA_IN2 = 40.0  # Lowered from 55: more permissive for far-wall robots
    BALL_FILTER_ENABLED = True
    # Scoring-element colors come from season.py (BIOBUZZ: yellow POLLEN).
    BALL_HSV_RANGES = season.BALL_HSV_RANGES
    BALL_MIN_FIELD_AREA_IN2 = 2.0
    # Shot-detection ball mask: a pixel counts as a moving ball when it has
    # the ball color AND differs from the median background by this much.
    BALL_MOTION_THRESH = 25
    BALL_SHOT_CLOSE_PX = 3   # small fixed close kernel; 0/1 disables
    # Perspective robot scale is clamped to this range (x center-field size).
    LOCAL_SCALE_MIN_FRAC = 0.35
    LOCAL_SCALE_MAX_FRAC = 3.0
    # BLOB_MIN shrinks for far robots, but never below this fraction of it.
    BLOB_MIN_SCALE_FLOOR = 0.2
    # Floor-contact positions up to this far outside the field are clamped
    # onto the edge instead of being dropped (only with FOOT_OFFSET_FRAC > 0).
    FIELD_EDGE_TOL_IN = 14.0
    BALL_MASK_OPEN_FRAC = 0.030
    BALL_MASK_CLOSE_FRAC = 0.055
    BALL_MASK_DILATE_FRAC = 0.070
    INIT_MAX_CANDIDATES = 8
    INIT_MIN_SPACING_IN = 8.0
    REID_HIST_H_BINS = 18
    REID_HIST_S_BINS = 16
    REID_SAMPLE_STRIDE = 15
    REID_MAX_SAMPLES_PER_ROBOT = 48
    REID_CROP_SCALE = 0.55
    REID_COST_WEIGHT = 1.60
    REID_COST_WEIGHT_REACQ = 2.70
    POST_MERGE_LOCK_FRAMES = 12
    POST_MERGE_LOCK_WEIGHT = 1.25
    POST_MERGE_LOCK_REJECT_PX = 115.0
    POST_MERGE_LOCK_APPEAR_WEIGHT = 1.55
    POST_MERGE_DUPLICATE_PX = 32.0
    POST_MERGE_DUPLICATE_IN = 12.0
    # Automatic appearance re-ID (season.AUTO_REID)
    AUTO_REID_ISOLATION_IN = 30.0   # no other visible robot this close
    AUTO_REID_STRIDE = 3            # keep every 3rd isolated sample
    AUTO_REID_EARLY_SAMPLES = 20    # first samples are kept for good
    AUTO_REID_RECENT_SAMPLES = 20   # plus a rolling window of recent ones
    AUTO_REID_MIN_SAMPLES = 5
    AUTO_REID_ADD_MAX_DIST = 0.5
    # Identity correction (season.APPEARANCE_ID_CORRECTION)
    IDFIX_MIN_EARLY = 8        # early samples needed before a robot's look is trusted
    IDFIX_EMA = 0.25           # smoothing of per-track appearance evidence
    IDFIX_MIN_OBS = 6          # isolated observations before a track's evidence counts
    IDFIX_MAX_AGE_S = 2.0      # evidence older than this is ignored
    IDFIX_MARGIN = 0.08        # required cost improvement per relabeled robot
    IDFIX_VOTES = 6            # consecutive frames the same relabeling must win

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self._bg           = None
        self._field_mask   = None
        self._H_2d         = None
        self._H_inv        = None
        self._kern         = cv2.getStructuringElement(
                                 cv2.MORPH_ELLIPSE, (self.KERNEL_PX, self.KERNEL_PX))
        self._neck_kern    = None
        self._robot_max_px = 60

        self.tracked_poses = [RobotPose() for _ in range(4)]
        self._pos          = [None] * 4
        self._pos_px       = [None] * 4
        self._vel          = [(0.0, 0.0)] * 4
        self._vel_px       = [(0.0, 0.0)] * 4
        self._coast        = [999]  * 4
        self._merge_recent = [0]    * 4
        self._initialized  = False
        self._track_features = [None] * 4

        self._merge_groups: Dict[FrozenSet, MergeGroup] = {}
        self._underresolved_tracks = set()
        self._post_merge_locks = [None] * 4
        self._last_ball_mask = None            # moving-ball mask (shot detection)
        self._last_ball_suppress_mask = None   # ball mask removed from robot FG
        self._shot_region_mask = None
        self.hive_boxes: Dict[str, object] = {}   # color -> int32 polygon (N,1,2)
        self.exclude_polys: List[object] = []
        self._init_wait_frames = 0
        self._init_history: List[List[Tuple[float, float]]] = []
        # Typical single-robot blob area / (18 in robot size)^2; learned at
        # initialization when season.AUTO_ROBOT_AREA is on.
        self._single_area_ratio = 1.0
        self._field_mask_open = None   # field mask without HIVE boxes removed
        self._hive_bg: Dict[str, dict] = {}
        self._hive_state: Dict[str, tuple] = {}
        # Time awareness (seconds, from update(time_s=...)) and auto re-ID.
        self._now: Optional[float] = None
        self._last_seen_t: List[Optional[float]] = [None] * 4
        self._auto_refs_early: List[List] = [[] for _ in range(4)]
        self._auto_refs_recent: List[List] = [[] for _ in range(4)]
        self._auto_ref_counter = [0] * 4
        # Identity correction (season.APPEARANCE_ID_CORRECTION)
        self._app_ema: List[Optional[object]] = [None] * 4
        self._app_obs = [0] * 4
        self._app_last_t: List[Optional[float]] = [None] * 4
        self._idfix_votes = 0
        self._idfix_key = None
        self._idfix_frame = 0
        self._reid_refs: Optional[Dict[int, List]] = None
        # Static blob suppression: track per-blob pixel location history
        # Format: {approx_pixel_key: consecutive_static_frames}
        self._blob_static_counts: dict = {}
        self._blob_static_positions: dict = {}
        # How many frames a blob must stay within STATIC_BLOB_MOVE_PX to be suppressed
        self.STATIC_BLOB_SUPPRESS_FRAMES = 20
        # If blob moves less than this many pixels, count it as static
        self.STATIC_BLOB_MOVE_PX = 8

        # Corner structure exclusion zone (field coords, inches).
        # The physical corner posts create persistent FG blobs that trap R0/R3.
        # Suppress blobs inside these corner zones unless a track is actively moving.
        # Zone: x < CORNER_ZONE_IN  for y < CORNER_ZONE_IN  (top-left and top-right)
        self.CORNER_ZONE_IN = 22.0   # 22-inch square at each top corner
        self._frame_counter = 0      # for corner zone timing

    # ── setup ────────────────────────────────────────────────────────────

    def setup(self, video_path, ordered_corners, frame_shape, calibration_extras=None):
        """
        calibration_extras (optional, from field_corners.json):
          hive_boxes_px : {"red": [[x,y],...], "blue": [[x,y],...]}
                          image polygons around each HIVE (scoring targets).
                          Also removed from robot detection.
          exclude_px    : [[[x,y],...], ...] image polygons ignored entirely
                          (on-screen clock, overlays, ...).
        """
        cv2, np = self.cv2, self.np
        h, w = frame_shape[:2]
        extras = calibration_extras or {}
        self.hive_boxes = {}
        for color, pts in (extras.get("hive_boxes_px") or {}).items():
            if pts and len(pts) >= 3:
                self.hive_boxes[str(color)] = np.array(
                    pts, dtype=np.int32).reshape(-1, 1, 2)
        self.exclude_polys = [
            np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            for pts in (extras.get("exclude_px") or []) if pts and len(pts) >= 3
        ]
        if self.hive_boxes:
            print("[INFO] HIVE boxes from calibration: {}".format(
                ", ".join(sorted(self.hive_boxes))))
        if self.exclude_polys:
            print("[INFO] Exclusion areas from calibration: {}".format(len(self.exclude_polys)))

        tl, tr, br, bl = ordered_corners
        cx_poly = (tl[0]+tr[0]+br[0]+bl[0]) / 4
        cy_poly = (tl[1]+tr[1]+br[1]+bl[1]) / 4
        # Tightened SIDE_PAD (10 vs 25) to reduce crowd/alliance-member bleed-in.
        # TOP_PAD increased (12 vs 5) to capture robots right at the far wall.
        # BOTTOM_PAD reduced (18 vs 25) to trim scoreboard noise.
        SIDE_PAD = 10; BOTTOM_PAD = 18; TOP_PAD = 12

        def _pad(x, y):
            dx = x - cx_poly; dy = y - cy_poly
            return (int(x + (SIDE_PAD   if dx > 0 else -SIDE_PAD)),
                    int(y + (BOTTOM_PAD if dy > 0 else -TOP_PAD)))

        poly = np.array([_pad(*tl), _pad(*tr), _pad(*br), _pad(*bl)], dtype=np.int32)
        self._field_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._field_mask, [poly], 255)

        # Region where airborne scoring elements are searched for: the field
        # plus the space above the far wall where the scoring targets are.
        shot_region = self._field_mask.copy()
        p_tl, p_tr = _pad(*tl), _pad(*tr)
        if self.hive_boxes:
            hull_pts = [p_tl, p_tr]
            for box in self.hive_boxes.values():
                hull_pts.extend([tuple(p) for p in box.reshape(-1, 2)])
                cv2.fillPoly(shot_region, [box], 255)
            hull = cv2.convexHull(np.array(hull_pts, dtype=np.int32).reshape(-1, 1, 2))
            cv2.fillPoly(shot_region, [hull], 255)
        else:
            above = np.array([p_tl, p_tr, (p_tr[0], 0), (p_tl[0], 0)], dtype=np.int32)
            cv2.fillPoly(shot_region, [above], 255)
        for ex in self.exclude_polys:
            cv2.fillPoly(shot_region, [ex], 0)
        self._shot_region_mask = shot_region

        # Exclusion areas (clock, overlays) are never used. HIVE boxes are
        # removed from robot detection too, unless the per-position HIVE
        # background can model them (see _foreground_mask).
        for poly_ex in self.exclude_polys:
            cv2.fillPoly(self._field_mask, [poly_ex], 0)
        self._field_mask_open = self._field_mask.copy()
        for poly_ex in self.hive_boxes.values():
            cv2.fillPoly(self._field_mask, [poly_ex], 0)

        dst2d = np.array([[0,0],[144,0],[144,144],[0,144]], dtype=np.float32)
        self._H_2d, _ = cv2.findHomography(ordered_corners, dst2d)
        self._H_inv   = np.linalg.inv(self._H_2d)

        c_f = np.array([[[72.0, 72.0]]], dtype=np.float32)
        r_f = np.array([[[72.0 + self.ROBOT_SIZE_IN, 72.0]]], dtype=np.float32)
        u_f = np.array([[[72.0, 72.0 + self.ROBOT_SIZE_IN]]], dtype=np.float32)
        c_px = cv2.perspectiveTransform(c_f, self._H_inv)[0][0]
        r_px = cv2.perspectiveTransform(r_f, self._H_inv)[0][0]
        u_px = cv2.perspectiveTransform(u_f, self._H_inv)[0][0]
        px_x = math.hypot(r_px[0]-c_px[0], r_px[1]-c_px[1])
        px_y = math.hypot(u_px[0]-c_px[0], u_px[1]-c_px[1])
        self._robot_max_px = max(px_x, px_y)
        print("[INFO] Robot pixel footprint: {:.1f} px / 18 in".format(self._robot_max_px))

        neck_r = max(3, int(self._robot_max_px * 0.15))
        self._neck_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (neck_r*2+1, neck_r*2+1))
        print("[INFO] Neck-breaking kernel radius: {} px".format(neck_r))

        cap2 = cv2.VideoCapture(video_path)
        n_total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        indices  = np.linspace(0, n_total - 1, self.N_BG_SAMPLES, dtype=int)
        bar = _make_bar("Building background", len(indices))
        frames = []
        for idx in indices:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, f = cap2.read()
            if ret:
                frames.append(f.astype(np.float32))
            bar.next()
        bar.finish()
        cap2.release()

        self._bg = (np.median(frames, axis=0).astype(np.uint8)
                    if frames else np.zeros((h, w, 3), dtype=np.uint8))
        self._build_hive_backgrounds(frames)
        print("[INFO] Background ready.")

    def _build_hive_backgrounds(self, frames) -> None:
        """Median background of each HIVE box for each of its two resting
        positions (frames split by the measured HIVE color level)."""
        self._hive_bg = {}
        if not season.HIVE_STATE_BACKGROUND or not self.hive_boxes or not frames:
            return
        from autoscout.hive import hive_box_bounds, measure_hive_level
        np = self.np
        h, w = self._bg.shape[:2]
        for color, poly in self.hive_boxes.items():
            ranges = season.HIVE_HSV_RANGES.get(color)
            if not ranges:
                continue
            box = hive_box_bounds(poly, w, h)
            levels = []
            for f in frames:
                lvl = measure_hive_level(self.cv2, np, f.astype(np.uint8), box, ranges)
                levels.append(lvl)
            valid = [lv for lv in levels if lv is not None]
            if len(valid) < 6:
                continue
            lo, hi = min(valid), max(valid)
            centers = [lo, hi] if hi - lo >= 0.08 else [float(np.median(valid))]
            for _ in range(10):   # 2-means on the levels
                groups = [[] for _ in centers]
                for lv in valid:
                    groups[min(range(len(centers)), key=lambda k: abs(lv - centers[k]))].append(lv)
                centers = [float(np.mean(g)) if g else c for g, c in zip(groups, centers)]
            x0, y0, x1, y1 = box
            states = []
            for center in centers:
                members = [f for f, lv in zip(frames, levels)
                           if lv is not None and abs(lv - center) <= 0.05]
                if len(members) < 4:
                    continue
                patch = np.median([m[y0:y1, x0:x1] for m in members], axis=0).astype(np.uint8)
                states.append((center, patch))
            if states:
                self._hive_bg[color] = {"box": box, "states": states}
                print("[INFO] {} HIVE background learned for {} position(s).".format(color, len(states)))

    def set_hive_state(self, state: Optional[Dict[str, tuple]]) -> None:
        """{color: (level or None, moving)} from the HIVE tip detector."""
        self._hive_state = dict(state or {})

    def _background_and_mask(self):
        """Background image and field mask for this frame: HIVE boxes use the
        background of the HIVE's current position when it is known and still;
        otherwise the box is removed from robot detection."""
        if not self._hive_bg or self._field_mask_open is None:
            return self._bg, self._field_mask
        bg = None
        mask = self._field_mask_open
        for color, poly in self.hive_boxes.items():
            info = self._hive_bg.get(color)
            level, moving = self._hive_state.get(color, (None, True))
            patch = None
            if info is not None and level is not None and not moving:
                center, best = min(info["states"], key=lambda st: abs(level - st[0]))
                if abs(level - center) <= 0.05:
                    patch = best
            if patch is None:
                if mask is self._field_mask_open:
                    mask = mask.copy()
                self.cv2.fillPoly(mask, [poly], 0)
                continue
            if bg is None:
                bg = self._bg.copy()
            x0, y0, x1, y1 = info["box"]
            bg[y0:y1, x0:x1] = patch
        return (bg if bg is not None else self._bg), mask

    # ── perspective helpers ───────────────────────────────────────────────

    def _footprint_px_at_field(self, fx: float, fy: float) -> float:
        """Pixel size of an 18 in robot footprint centered at field (fx, fy)."""
        np = self.np
        half = self.ROBOT_SIZE_IN / 2.0
        pts = np.array([[
            [fx - half, fy], [fx + half, fy],
            [fx, fy - half], [fx, fy + half],
        ]], dtype=np.float32)
        img = self.cv2.perspectiveTransform(pts, self._H_inv)[0]
        sx = math.hypot(img[1][0] - img[0][0], img[1][1] - img[0][1])
        sy = math.hypot(img[3][0] - img[2][0], img[3][1] - img[2][1])
        return max(sx, sy)

    def _clamped_scale(self, px: float) -> float:
        base = max(self._robot_max_px, 1.0)
        return min(max(px, base * self.LOCAL_SCALE_MIN_FRAC),
                   base * self.LOCAL_SCALE_MAX_FRAC)

    def _foot_offset_px(self, u: float, v: float) -> float:
        frac = season.FOOT_OFFSET_FRAC
        if frac <= 0 or self._H_2d is None:
            return 0.0
        if season.PERSPECTIVE_ROBOT_SCALE:
            fp = self._raw_image_to_field(u, v)
            fx = min(max(fp[0], 0.0), 144.0)
            fy = min(max(fp[1], 0.0), 144.0)
            scale = self._clamped_scale(self._footprint_px_at_field(fx, fy))
        else:
            scale = self._robot_max_px
        return frac * scale

    def _raw_image_to_field(self, u: float, v: float) -> Tuple[float, float]:
        pt = self.np.array([[[u, v]]], dtype=self.np.float32)
        fp = self.cv2.perspectiveTransform(pt, self._H_2d)[0][0]
        return float(fp[0]), float(fp[1])

    def _image_to_field(self, u: float, v: float, bottom_v: Optional[float] = None) -> Tuple[float, float]:
        """
        Project a blob/peak center to the field, using the robot's estimated
        floor-contact point (center moved down by FOOT_OFFSET_FRAC * local
        robot size, never below the blob's bottom edge).
        """
        foot_v = v + self._foot_offset_px(u, v)
        if bottom_v is not None:
            foot_v = max(v, min(foot_v, float(bottom_v)))
        return self._raw_image_to_field(u, foot_v)

    def field_to_image(self, fx: float, fy: float) -> Tuple[float, float]:
        """Field position (corner-origin inches) -> image point at the robot's
        visual center (inverse of _image_to_field)."""
        pt = self.np.array([[[fx, fy]]], dtype=self.np.float32)
        ip = self.cv2.perspectiveTransform(pt, self._H_inv)[0][0]
        u, v = float(ip[0]), float(ip[1])
        frac = season.FOOT_OFFSET_FRAC
        if frac > 0:
            if season.PERSPECTIVE_ROBOT_SCALE:
                scale = self._clamped_scale(self._footprint_px_at_field(
                    min(max(fx, 0.0), 144.0), min(max(fy, 0.0), 144.0)))
            else:
                scale = self._robot_max_px
            v -= frac * scale
        return u, v

    def _robot_px_at(self, u: float, v: float) -> float:
        """18 in robot size in pixels for a robot whose visual center is at
        image (u, v). Falls back to the single center-field size."""
        if not season.PERSPECTIVE_ROBOT_SCALE or self._H_2d is None:
            return self._robot_max_px
        fx, fy = self._image_to_field(u, v)
        fx = min(max(fx, 0.0), 144.0)
        fy = min(max(fy, 0.0), 144.0)
        return self._clamped_scale(self._footprint_px_at_field(fx, fy))

    def _contour_robot_px(self, contour) -> float:
        if contour is None or not season.PERSPECTIVE_ROBOT_SCALE:
            return self._robot_max_px
        M = self.cv2.moments(contour)
        if M["m00"] <= 0:
            return self._robot_max_px
        return self._robot_px_at(M["m10"] / M["m00"], M["m01"] / M["m00"])

    def _track_robot_px(self, track_id: int) -> float:
        if not season.PERSPECTIVE_ROBOT_SCALE:
            return self._robot_max_px
        if self._pos[track_id] is None and self._pos_px[track_id] is None:
            return self._robot_max_px
        px, py = self._predict_image_pos(track_id)
        return self._robot_px_at(px, py)

    def _blob_min_area(self, contour) -> float:
        if not season.PERSPECTIVE_ROBOT_SCALE:
            return float(self.BLOB_MIN)
        ratio = self._contour_robot_px(contour) / max(self._robot_max_px, 1.0)
        return self.BLOB_MIN * min(1.0, max(ratio * ratio, self.BLOB_MIN_SCALE_FLOOR))

    # ── main update ───────────────────────────────────────────────────────

    def update(self, frame, _H=None, time_s: Optional[float] = None):
        self._now = time_s
        if self._bg is None or self._H_2d is None:
            for p in self.tracked_poses:
                p.visible = False
            return self.tracked_poses

        for i in range(4):
            if self._merge_recent[i] > 0:
                self._merge_recent[i] -= 1
            lock = self._post_merge_locks[i]
            if lock is not None:
                lock["frames_left"] -= 1
                if lock["frames_left"] <= 0:
                    self._post_merge_locks[i] = None
                else:
                    img_pos = lock["img_pos"]
                    img_vel = lock["img_vel"]
                    field_pos = lock["field_pos"]
                    field_vel = lock["field_vel"]
                    lock["img_pos"] = (img_pos[0] + img_vel[0], img_pos[1] + img_vel[1])
                    lock["field_pos"] = (field_pos[0] + field_vel[0], field_pos[1] + field_vel[1])

        blobs = self._get_blobs(frame)
        self._underresolved_tracks = self._find_underresolved_tracks(blobs)

        if not self._initialized:
            self._init_wait_frames += 1
            init_blobs = self._stable_init_blobs(blobs)
            lineup = self._select_bootstrap_lineup(init_blobs)
            reason = "best visible 4-blob lineup"
            if (lineup is None
                    and self._init_wait_frames >= season.INIT_PARTIAL_AFTER_FRAMES
                    and len({b[3] for b in init_blobs}) >= season.INIT_MIN_ROBOTS):
                # Some robots are hidden (e.g. behind the HIVE). Start with the
                # visible ones; the others are picked up when they appear.
                count = min(len({b[3] for b in init_blobs}), 4)
                lineup = self._select_bootstrap_lineup(init_blobs, count=count)
                reason = "partial {}-blob lineup ({} robot(s) not visible yet)".format(
                    count, 4 - count)
            if lineup is None:
                for p in self.tracked_poses:
                    p.visible = False
                return self.tracked_poses
            self._learn_single_robot_area(init_blobs)
            self._initialize_tracks_from_lineup(lineup, reason)
            return self.tracked_poses

        assignment = self._assign(blobs)

        pid_to_contour = {}
        for b in blobs:
            if b[3] not in pid_to_contour:
                pid_to_contour[b[3]] = b[8]

        pid_to_tracks = defaultdict(list)
        for ti, bi in assignment.items():
            pid_to_tracks[blobs[bi][3]].append(ti)
        merged_pids = {pid: tl for pid, tl in pid_to_tracks.items() if len(tl) > 1}

        active_keys = set()
        for pid, tlist in merged_pids.items():
            key = frozenset(tlist)
            active_keys.add(key)
            if key not in self._merge_groups:
                self._merge_groups[key] = self._create_merge_group(tlist, pid)
                print("[INFO] Merge started: tracks {}".format(sorted(tlist)))
            mg = self._merge_groups[key]
            mg.parent_id = pid
            if pid in pid_to_contour:
                self._update_crossing(mg, pid_to_contour[pid])
            for tid in tlist:
                self._merge_recent[tid] = self.MERGE_HOLD

        for key in list(self._merge_groups.keys()):
            if key not in active_keys:
                mg = self._merge_groups.pop(key)
                if self._uses_multi_merge(mg) and self._relabel_multi_merge_exit(mg, blobs, assignment):
                    for tid in mg.track_ids:
                        self._merge_recent[tid] = self.MERGE_HOLD
                else:
                    self._apply_separation(mg)

        self._prune_post_merge_duplicate_assignments(blobs, assignment)

        # tracks inside a 2-robot merge have positions frozen (handled below)
        # tracks inside a 3+ robot merge get positions from peak_assignment
        two_merged_tracks = set()
        multi_merged_peaks = {}
        for mg in self._merge_groups.values():
            if not self._uses_multi_merge(mg):
                two_merged_tracks.update(mg.track_ids)
            else:
                # FIX (e): use continuously-updated peak positions for 3+ merges
                multi_merged_peaks.update(mg.peak_assignment)

        for i in range(4):
            if i in multi_merged_peaks:
                px, py = multi_merged_peaks[i]
                fx, fy = self._image_to_field(px, py)
                self._coast[i] = 0
                self._last_seen_t[i] = self._now
                self.tracked_poses[i].x_in = fx
                self.tracked_poses[i].y_in = fy
                self.tracked_poses[i].visible = True
                self._update_track_motion(i, (fx, fy), (px, py))
                self._update_post_merge_lock(i, (fx, fy), (px, py))
                self.tracked_poses[i].heading = self._motion_heading(
                    i, self.tracked_poses[i].heading)
                self._pos[i] = (fx, fy)
                self._pos_px[i] = (px, py)
                continue
            if i in assignment:
                fx, fy, _hdg, _pid, cx, cy, _qual, _nsplit, _cnt, *_rest = blobs[assignment[i]]
                self._coast[i] = 0
                self.tracked_poses[i].x_in    = fx
                self.tracked_poses[i].y_in    = fy
                self.tracked_poses[i].visible = True
                self._last_seen_t[i] = self._now
                if i not in two_merged_tracks:
                    self._update_track_motion(i, (fx, fy), (cx, cy))
                    feature = blobs[assignment[i]][9] if len(blobs[assignment[i]]) > 9 else None
                    self._track_features[i] = feature
                    self._maybe_add_auto_ref(i, blobs, assignment)
                    self._update_post_merge_lock(i, (fx, fy), (cx, cy))
                    self.tracked_poses[i].heading = self._motion_heading(
                        i, self.tracked_poses[i].heading)
                    self._pos[i]    = (fx, fy)
                    self._pos_px[i] = (cx, cy)
                else:
                    self.tracked_poses[i].heading = self._motion_heading(
                        i, self.tracked_poses[i].heading)
            else:
                self._coast[i] = min(self._coast[i] + 1, self.MAX_COAST + 1)
                self.tracked_poses[i].visible = (
                    self._coast[i] <= self.VISIBLE_COAST
                    and self._pos[i] is not None)
                if self._pos[i] is not None:
                    self.tracked_poses[i].x_in = self._pos[i][0]
                    self.tracked_poses[i].y_in = self._pos[i][1]
                    self.tracked_poses[i].heading = self._motion_heading(
                        i, self.tracked_poses[i].heading)

        # Appearance evidence and identity correction run last, after all
        # positions for this frame are written.
        self._update_identity_evidence(blobs, assignment)
        return self.tracked_poses

    def _update_post_merge_lock(self, track_id: int, field_pos, image_pos) -> None:
        lock = self._post_merge_locks[track_id]
        if lock is None:
            return
        prev_img = lock["img_pos"]
        prev_field = lock["field_pos"]
        lock["img_vel"] = (float(image_pos[0]) - prev_img[0], float(image_pos[1]) - prev_img[1])
        lock["field_vel"] = (float(field_pos[0]) - prev_field[0], float(field_pos[1]) - prev_field[1])
        lock["img_pos"] = (float(image_pos[0]), float(image_pos[1]))
        lock["field_pos"] = (float(field_pos[0]), float(field_pos[1]))

    def _prime_post_merge_lock(self, track_id: int, field_pos, image_pos,
                               prev_image_pos=None, feature=None,
                               merge_group_key=None) -> None:
        prev_field = self._pos[track_id] if self._pos[track_id] is not None else field_pos
        if prev_image_pos is None:
            img_vel = (0.0, 0.0)
        else:
            img_vel = (
                float(image_pos[0]) - float(prev_image_pos[0]),
                float(image_pos[1]) - float(prev_image_pos[1]),
            )
        field_vel = (
            float(field_pos[0]) - float(prev_field[0]),
            float(field_pos[1]) - float(prev_field[1]),
        )
        self._post_merge_locks[track_id] = {
            "frames_left": self.POST_MERGE_LOCK_FRAMES,
            "img_pos": (float(image_pos[0]), float(image_pos[1])),
            "img_vel": img_vel,
            "field_pos": (float(field_pos[0]), float(field_pos[1])),
            "field_vel": field_vel,
            "feature": feature if feature is not None else self._track_features[track_id],
            "merge_group_key": merge_group_key,
        }

    def _assignment_support_cost(self, track_id: int, blob) -> float:
        pred_field = self._predict_field_pos(track_id)
        pred_px_x, pred_px_y = self._predict_image_pos(track_id)
        d_field = math.hypot(float(blob[0]) - pred_field[0], float(blob[1]) - pred_field[1])
        d_img = math.hypot(float(blob[4]) - pred_px_x, float(blob[5]) - pred_px_y)
        robot_px = self._track_robot_px(track_id)
        cost = (
            d_field / max(self.MAX_REACQ_IN, 1.0)
            + 0.35 * d_img / max(robot_px * 1.25, 1.0)
            + float(blob[6])
            + self._appearance_cost(track_id, blob)
        )
        lock = self._post_merge_locks[track_id]
        if lock is not None:
            lock_img_x, lock_img_y = lock["img_pos"]
            d_lock_img = math.hypot(float(blob[4]) - lock_img_x, float(blob[5]) - lock_img_y)
            cost += self.POST_MERGE_LOCK_WEIGHT * (
                d_lock_img / max(robot_px, 1.0))
            cost += self.POST_MERGE_LOCK_APPEAR_WEIGHT * self._feature_distance(
                lock.get("feature"),
                blob[9] if len(blob) > 9 else None,
            )
        return cost

    def _prune_post_merge_duplicate_assignments(self, blobs, assignment: Dict[int, int]) -> None:
        if len(assignment) < 2:
            return

        removed = set()
        assigned_tracks = list(assignment.keys())
        for idx, tid_a in enumerate(assigned_tracks):
            if tid_a in removed or tid_a not in assignment:
                continue
            lock_a = self._post_merge_locks[tid_a]
            if lock_a is None or lock_a.get("merge_group_key") is None:
                continue
            blob_a = blobs[assignment[tid_a]]
            for tid_b in assigned_tracks[idx + 1:]:
                if tid_b in removed or tid_b not in assignment:
                    continue
                lock_b = self._post_merge_locks[tid_b]
                if lock_b is None or lock_b.get("merge_group_key") != lock_a.get("merge_group_key"):
                    continue
                blob_b = blobs[assignment[tid_b]]
                d_img = math.hypot(float(blob_a[4]) - float(blob_b[4]),
                                   float(blob_a[5]) - float(blob_b[5]))
                d_field = math.hypot(float(blob_a[0]) - float(blob_b[0]),
                                     float(blob_a[1]) - float(blob_b[1]))
                if d_img > self.POST_MERGE_DUPLICATE_PX or d_field > self.POST_MERGE_DUPLICATE_IN:
                    continue

                cost_a = self._assignment_support_cost(tid_a, blob_a)
                cost_b = self._assignment_support_cost(tid_b, blob_b)
                drop_tid = tid_a if cost_a > cost_b else tid_b
                removed.add(drop_tid)

        for tid in removed:
            del assignment[tid]
            self._post_merge_locks[tid] = None
            self._coast[tid] = self.VISIBLE_COAST
            self.tracked_poses[tid].visible = False
            print("[INFO] Pruned duplicate post-merge track: R{}".format(tid))

    def _initialize_tracks_from_lineup(self, lineup, reason: str):
        for i, (fx, fy, cx, cy) in enumerate(lineup):
            self._pos[i] = (fx, fy)
            self._pos_px[i] = (cx, cy)
            self._vel[i] = (0.0, 0.0)
            self._vel_px[i] = (0.0, 0.0)
            self._coast[i] = 0
            self.tracked_poses[i].x_in = fx
            self.tracked_poses[i].y_in = fy
            self.tracked_poses[i].heading = 0.0
            self.tracked_poses[i].visible = True
        self._initialized = True
        print("[INFO] Tracker initialized from {}.".format(reason))

    def _learn_single_robot_area(self, blobs) -> None:
        if not season.AUTO_ROBOT_AREA:
            return
        ratios = []
        seen = set()
        for b in blobs:
            if b[3] in seen or b[7] != 1 or b[8] is None:
                continue
            seen.add(b[3])
            px = self._contour_robot_px(b[8])
            ratios.append(float(self.cv2.contourArea(b[8])) / max(px * px, 1.0))
        if len(ratios) < 2:
            return
        ratios.sort()
        median = ratios[len(ratios) // 2]
        self._single_area_ratio = min(max(median, 0.3), 1.2)
        print("[INFO] Single-robot blob size: {:.2f}x an 18 in square (from {} robots).".format(
            self._single_area_ratio, len(ratios)))

    def _stable_init_blobs(self, blobs):
        n = max(1, int(season.INIT_STABLE_FRAMES))
        self._init_history.append([(float(b[0]), float(b[1])) for b in blobs])
        self._init_history = self._init_history[-n:]
        if n <= 1:
            return blobs
        if len(self._init_history) < n:
            return []
        tol = season.INIT_STABLE_IN
        return [
            b for b in blobs
            if all(
                any(math.hypot(b[0] - x, b[1] - y) <= tol for x, y in frame_pts)
                for frame_pts in self._init_history
            )
        ]

    def _select_bootstrap_lineup(self, blobs, count: int = 4):
        if count <= 0 or len(blobs) < count:
            return None

        candidate_pool = blobs[:min(len(blobs), self.INIT_MAX_CANDIDATES)]
        best = None
        best_score = float("inf")
        for combo in itertools.combinations(candidate_pool, count):
            # Each starting robot must come from its own foreground blob; a
            # single blob split into pieces (robot + a person's arm, ...) must
            # not seed several tracks.
            if len({b[3] for b in combo}) < count:
                continue
            lineup = sorted(combo, key=lambda b: b[0])
            spacing_penalty = 0.0
            for left, right in zip(lineup, lineup[1:]):
                gap = right[0] - left[0]
                if gap < self.INIT_MIN_SPACING_IN:
                    spacing_penalty += (self.INIT_MIN_SPACING_IN - gap) ** 2

            score = sum(b[6] for b in lineup) + 0.04 * spacing_penalty
            if score < best_score:
                best_score = score
                best = [(float(b[0]), float(b[1]), float(b[4]), float(b[5]))
                        for b in lineup]
        return best

    # ── optimal assignment ────────────────────────────────────────────────

    def _assign(self, blobs: list) -> Dict[int, int]:
        np = self.np
        n = len(blobs)

        INF       = 1e9
        SKIP_COST = 2.0

        seeded = [t for t in range(4) if self._pos[t] is not None]
        real_cost = np.full((4, max(n, 1)), INF, dtype=np.float64)
        for i in range(4):
            if self._pos[i] is None:
                # Track never seen (robot hidden at start, e.g. behind the
                # HIVE). It may only claim a clean, unsplit blob that is well
                # away from every known robot.
                for j, b in enumerate(blobs):
                    if b[7] != 1 or b[6] >= 0.9:
                        continue
                    far = all(
                        math.hypot(b[0] - self._pos[t][0], b[1] - self._pos[t][1])
                        >= self.ROBOT_SIZE_IN * 1.5
                        for t in seeded
                    )
                    if far:
                        real_cost[i, j] = 0.5 + b[6]
                continue
            pred_x, pred_y = self._predict_field_pos(i)
            pred_px_x, pred_px_y = self._predict_image_pos(i)
            robot_px = self._track_robot_px(i)
            lost = self._coast[i] > self.VISIBLE_COAST
            visible_others = [
                self._pos[t] for t in seeded
                if t != i and self._coast[t] <= self.VISIBLE_COAST
            ] if (lost and season.PROTECT_VISIBLE_FROM_LOST) else []
            # A robot cannot have moved further than it can drive since it was
            # last seen (applies to every track, not only lost ones).
            reach = None
            if (season.LOST_TRACK_RULES and self._now is not None
                    and self._last_seen_t[i] is not None):
                reach = (season.LOST_REACQ_SPEED_IN_S * max(0.0, self._now - self._last_seen_t[i])
                         + season.LOST_REACQ_SLACK_IN)
            for j, b in enumerate(blobs):
                if lost and season.LOST_TRACK_RULES and b[7] != 1:
                    continue
                if reach is not None and math.hypot(b[0] - self._pos[i][0], b[1] - self._pos[i][1]) > reach:
                    continue
                if visible_others and b[7] == 1 and any(
                        math.hypot(b[0] - ox, b[1] - oy) < self.ROBOT_SIZE_IN
                        for ox, oy in visible_others):
                    continue
                d_field = math.hypot(b[0]-pred_x, b[1]-pred_y)
                d_img   = math.hypot(b[4]-pred_px_x, b[5]-pred_px_y)
                if i in self._underresolved_tracks and not self._blob_matches_track_merge(i, b):
                    continue
                if not self._is_plausible_match(i, b, d_field, d_img):
                    continue
                lock = self._post_merge_locks[i]
                if lock is not None:
                    lock_img_x, lock_img_y = lock["img_pos"]
                    d_lock_img = math.hypot(b[4] - lock_img_x, b[5] - lock_img_y)
                    if d_lock_img > self.POST_MERGE_LOCK_REJECT_PX:
                        continue
                    lock_feature_cost = self._feature_distance(lock.get("feature"), b[9] if len(b) > 9 else None)
                else:
                    d_lock_img = 0.0
                    lock_feature_cost = 0.0
                dist_scale = (self.MAX_REACQ_IN
                              if self._coast[i] > self.VISIBLE_COAST
                              else self.MAX_DIST_IN)
                dist_cost = d_field / max(dist_scale, 1.0)
                img_cost  = d_img / max(robot_px * 1.25, 1.0)
                qual_cost = b[6]
                appearance_cost = self._appearance_cost(i, b)
                appearance_weight = (
                    self.REID_COST_WEIGHT_REACQ
                    if self._coast[i] > self.VISIBLE_COAST
                    else self.REID_COST_WEIGHT
                )
                real_cost[i, j] = (
                    dist_cost
                    + 0.35 * img_cost
                    + qual_cost
                    + appearance_weight * appearance_cost
                    + self.POST_MERGE_LOCK_WEIGHT * (
                        d_lock_img / max(robot_px, 1.0))
                    + self.POST_MERGE_LOCK_APPEAR_WEIGHT * lock_feature_cost
                )

        if n == 0:
            return {}

        skip_cols = np.full((4, 4), INF, dtype=np.float64)
        for i in range(4):
            if self._pos[i] is not None:
                coast_frac = min(self._coast[i] / max(self.MAX_COAST, 1), 1.0)
                skip_cost_i = SKIP_COST * (1.0 + 0.35 * coast_frac)
                if self._coast[i] > self.VISIBLE_COAST:
                    skip_cost_i = SKIP_COST * (1.8 + 0.7 * coast_frac)
                if self._merge_recent[i] > 0:
                    skip_cost_i = min(skip_cost_i, SKIP_COST * 0.8)
                if self._post_merge_locks[i] is not None:
                    skip_cost_i += 0.6
            else:
                skip_cost_i = SKIP_COST
            skip_cols[i, i] = skip_cost_i

        cost = np.hstack([real_cost, skip_cols])

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            assignment = {}
            for r, c in zip(row_ind, col_ind):
                if c < n:
                    assignment[r] = c
        except ImportError:
            assignment = {}
            work = cost.copy()
            total_cols = n + 4
            for _ in range(4):
                idx = int(np.argmin(work))
                ri  = idx // total_cols
                ci  = idx  % total_cols
                if work[ri, ci] >= INF:
                    break
                if ci < n:
                    assignment[ri] = ci
                work[ri, :] = INF
                work[:, ci] = INF

        # A never-seen track must not start inside another robot's blob.
        for i in [t for t in assignment if self._pos[t] is None]:
            pid = blobs[assignment[i]][3]
            if any(blobs[assignment[t]][3] == pid for t in assignment if t != i):
                del assignment[i]
            else:
                print("[INFO] Track {} acquired (robot was not visible at start).".format(i))

        return assignment

    def _blob_matches_track_merge(self, track_id: int, blob) -> bool:
        pred_px_x, pred_px_y = self._predict_image_pos(track_id)
        contour = blob[8]
        if contour is None:
            return False
        dist = abs(self.cv2.pointPolygonTest(contour, (float(pred_px_x), float(pred_px_y)), True))
        return dist <= self._track_robot_px(track_id) * 0.75

    def _predict_field_pos(self, track_id: int) -> Tuple[float, float]:
        if self._pos[track_id] is None:
            return (0.0, 0.0)
        px, py = self._pos[track_id]
        vx, vy = self._vel[track_id]
        coast = min(self._coast[track_id], self.MAX_COAST)
        horizon = min(1.0 + 0.15 * coast, 2.0)
        return (px + vx * horizon, py + vy * horizon)

    def _predict_image_pos(self, track_id: int) -> Tuple[float, float]:
        if self._pos_px[track_id] is None:
            if self._pos[track_id] is None:
                return (0.0, 0.0)
            return self.field_to_image(self._pos[track_id][0], self._pos[track_id][1])
        px, py = self._pos_px[track_id]
        vx, vy = self._vel_px[track_id]
        coast = min(self._coast[track_id], self.MAX_COAST)
        horizon = min(1.0 + 0.15 * coast, 2.0)
        return (px + vx * horizon, py + vy * horizon)

    def _update_track_motion(self, track_id: int, field_pos, image_pos) -> None:
        old_field = self._pos[track_id]
        old_image = self._pos_px[track_id]
        if (season.RESET_VELOCITY_ON_JUMP and old_field is not None
                and math.hypot(field_pos[0] - old_field[0],
                               field_pos[1] - old_field[1]) > self.MAX_DIST_IN):
            self._vel[track_id] = (0.0, 0.0)
            self._vel_px[track_id] = (0.0, 0.0)
            return
        if old_field is not None:
            raw_vx = field_pos[0] - old_field[0]
            raw_vy = field_pos[1] - old_field[1]
            speed = math.hypot(raw_vx, raw_vy)
            if speed > self.MAX_SPEED_IN:
                scale = self.MAX_SPEED_IN / max(speed, 1e-9)
                raw_vx *= scale
                raw_vy *= scale
            prev_vx, prev_vy = self._vel[track_id]
            self._vel[track_id] = (
                prev_vx * 0.45 + raw_vx * 0.55,
                prev_vy * 0.45 + raw_vy * 0.55,
            )
        if old_image is not None:
            raw_vx_px = image_pos[0] - old_image[0]
            raw_vy_px = image_pos[1] - old_image[1]
            prev_vx_px, prev_vy_px = self._vel_px[track_id]
            self._vel_px[track_id] = (
                prev_vx_px * 0.45 + raw_vx_px * 0.55,
                prev_vy_px * 0.45 + raw_vy_px * 0.55,
            )

    def _motion_heading(self, track_id: int, fallback: float = 0.0) -> float:
        vx, vy = self._vel[track_id]
        if math.hypot(vx, vy) < 0.25:
            return fallback
        return math.atan2(vy, vx)

    def _is_plausible_match(self, track_id: int, blob, d_field: float, d_img: float) -> bool:
        contour_count = max(blob[7], 1)
        effective_hold = max(self._coast[track_id], self._merge_recent[track_id])
        if effective_hold > 0 and d_field > self.MAX_REACQ_IN:
            return False
        img_limit = self._track_robot_px(track_id) * (
            1.3 + self.REACQ_PX_PAD * min(effective_hold, 4))
        if d_img > img_limit:
            return False
        if contour_count == 1 and blob[6] >= 0.9 and effective_hold > 0:
            return False
        if self._merge_recent[track_id] > 0 and contour_count == 1 and blob[6] >= 0.45:
            return False
        return True

    # ── merge group lifecycle ─────────────────────────────────────────────

    def _create_merge_group(self, track_ids: List[int], parent_id: int) -> MergeGroup:
        """
        Create a MergeGroup.  For 3+ tracks, record entry_order so we can
        detect permutations during the merge (fix f).
        """
        np = self.np
        positions = []
        for tid in track_ids:
            if self._pos_px[tid] is not None:
                positions.append((tid, float(self._pos_px[tid][0]),
                                       float(self._pos_px[tid][1])))
            elif self._pos[tid] is not None:
                ipx, ipy = self.field_to_image(self._pos[tid][0], self._pos[tid][1])
                positions.append((tid, ipx, ipy))
            else:
                positions.append((tid, 0.0, 0.0))

        # Principal axis: direction of maximum spread among the entry positions
        ax, ay = 1.0, 0.0
        best = 0.0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dx = positions[j][1] - positions[i][1]
                dy = positions[j][2] - positions[i][2]
                d  = math.hypot(dx, dy)
                if d > best:
                    best = d
                    ax, ay = dx/max(d, 1e-9), dy/max(d, 1e-9)

        cx = sum(p[1] for p in positions) / len(positions)
        cy = sum(p[2] for p in positions) / len(positions)

        def _proj(px, py): return (px-cx)*ax + (py-cy)*ay
        ordered_positions = sorted(positions, key=lambda p: _proj(p[1], p[2]))
        entry_order = [p[0] for p in ordered_positions]

        # Seed peak_assignment with entry positions so we have valid state
        # immediately (guards against first-frame stale-peak scenario, fix g)
        peak_assignment = {tid: (px, py) for tid, px, py in positions}
        order_votes = {
            tid: [0.0] * len(entry_order)
            for tid in track_ids
        }
        entry_features = {
            tid: self._track_features[tid]
            for tid in track_ids
            if self._track_features[tid] is not None
        }
        for slot, tid in enumerate(entry_order):
            order_votes[tid][slot] += 1.0

        return MergeGroup(
            track_ids      = list(track_ids),
            entry_axis     = (ax, ay),
            parent_id      = parent_id,
            crossed        = False,
            entry_order    = entry_order,
            current_order  = list(entry_order),   # starts as identity permutation
            peak_assignment= peak_assignment,
            order_votes    = order_votes,
            entry_features = entry_features,
        )

    def _uses_multi_merge(self, mg: MergeGroup) -> bool:
        return len(mg.track_ids) > 2 or season.FOLLOW_TWO_ROBOT_MERGES

    def _update_crossing(self, mg: MergeGroup, contour) -> None:
        """
        Update per-frame merge state.

        2-robot merges: unchanged from v3.1 (crossed flag).
        3+ robot merges: update peak_assignment AND current_order so we can
        detect the full permutation at separation (fixes e, f, g).
        """
        cv2 = self.cv2
        peaks = self._split_contour(contour)
        n = len(mg.track_ids)

        # FIX (g): if we got fewer peaks than tracks, keep existing peak_assignment
        # rather than silently doing nothing.  This preserves the last good state.
        if len(peaks) < 2:
            # Fully overlapping — can't distinguish positions; preserve state.
            return

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return
        mcx = M["m10"] / M["m00"]
        mcy = M["m01"] / M["m00"]
        ax, ay = mg.entry_axis

        def _proj(px, py): return (px - mcx) * ax + (py - mcy) * ay

        # ── 2-robot: original crossing logic (unchanged) ──────────────────
        if not self._uses_multi_merge(mg):
            p0, p1 = sorted(peaks[:2], key=lambda p: _proj(p[0], p[1]))
            dot = (p1[0] - p0[0]) * ax + (p1[1] - p0[1]) * ay
            mg.crossed = (dot < 0)
            return

        # ── 3+ robot: assign peaks to tracks then update current_order ────
        #
        # FIX (e) + (f): we assign each peak to the nearest track using the
        # CURRENT positions (which are themselves updated from peaks each frame
        # via the main update loop), not the frozen entry positions.  This
        # means prediction tracks the robots through the blob rather than
        # drifting back to entry.
        #
        # If peaks < n (fix g): assign what we can, leave the rest at their
        # last known positions.  current_order is only updated when we have
        # enough peaks to determine it.

        # Sort available peaks along entry axis
        sorted_peaks = sorted(peaks, key=lambda p: _proj(p[0], p[1]))

        # Build new assignment: for each track, find nearest unused peak
        new_assignment = dict(mg.peak_assignment)  # start from last good state

        # Use current positions (updated each frame from peaks) not entry positions
        track_current_px = []
        for tid in mg.track_ids:
            if self._pos_px[tid] is not None:
                track_current_px.append((tid, float(self._pos_px[tid][0]),
                                              float(self._pos_px[tid][1])))
            else:
                # Fall back to transformed field position
                if self._pos[tid] is not None:
                    ipx, ipy = self.field_to_image(self._pos[tid][0], self._pos[tid][1])
                    track_current_px.append((tid, ipx, ipy))
                else:
                    track_current_px.append((tid, mcx, mcy))

        # Sort tracks by current projection so nearby tracks compete fairly
        track_current_px.sort(key=lambda t: _proj(t[1], t[2]))
        successfully_assigned = 0
        used_peaks = set()
        for tid, tx, ty in track_current_px:
            best_d, best_k = float('inf'), None
            for k, (px, py) in enumerate(sorted_peaks):
                if k in used_peaks:
                    continue
                d = math.hypot(px - tx, py - ty)
                if d < best_d:
                    best_d, best_k = d, k
            if best_k is not None:
                new_assignment[tid] = sorted_peaks[best_k]
                used_peaks.add(best_k)
                successfully_assigned += 1

        mg.peak_assignment = new_assignment

        projected_order = sorted(
            mg.track_ids,
            key=lambda tid: _proj(*mg.peak_assignment[tid])
            if tid in mg.peak_assignment else 0.0
        )
        mg.current_order = projected_order
        self._record_merge_order(
            mg,
            projected_order,
            weight=1.0 if successfully_assigned >= n else 0.35,
        )

    def _record_merge_order(self, mg: MergeGroup, order: List[int], weight: float) -> None:
        if not mg.order_votes or not order:
            return
        for slot, tid in enumerate(order):
            if tid not in mg.order_votes:
                mg.order_votes[tid] = [0.0] * len(order)
            if slot < len(mg.order_votes[tid]):
                mg.order_votes[tid][slot] += weight

    def _resolve_merge_order(self, mg: MergeGroup) -> List[int]:
        if not mg.order_votes:
            return list(mg.current_order or mg.entry_order)

        best_order = None
        best_score = float("-inf")
        for order in itertools.permutations(mg.track_ids):
            score = 0.0
            for slot, tid in enumerate(order):
                score += mg.order_votes.get(tid, [0.0] * len(order))[slot]
            if score > best_score:
                best_score = score
                best_order = list(order)
        return best_order or list(mg.current_order or mg.entry_order)

    def _apply_separation(self, mg: MergeGroup) -> None:
        """
        Re-anchor track state on separation.

        2-robot merges: unchanged — swap if crossed.
        3+ robot merges: FIX (h) — apply the full permutation derived from
        entry_order vs current_order, then re-anchor each track to its
        current peak position.
        """
        for tid in mg.track_ids:
            self._merge_recent[tid] = self.MERGE_HOLD

        # ── 3+ robot case (and 2-robot with FOLLOW_TWO_ROBOT_MERGES) ──────
        if self._uses_multi_merge(mg):
            self._apply_separation_multi(mg)
            return

        # ── 2-robot case (unchanged from v3.1) ────────────────────────────
        if mg.crossed:
            tids = mg.track_ids
            n    = len(tids)
            sp   = [self._pos[tids[k]]    for k in range(n)]
            sppx = [self._pos_px[tids[k]] for k in range(n)]
            sv   = [self._vel[tids[k]]    for k in range(n)]
            svpx = [self._vel_px[tids[k]] for k in range(n)]
            for k in range(n):
                self._pos[tids[k]]    = sp[n-1-k]
                self._pos_px[tids[k]] = sppx[n-1-k]
                self._vel[tids[k]]    = sv[n-1-k]
                self._vel_px[tids[k]] = svpx[n-1-k]
            print("[INFO] Separation WITH crossing — swapped: {}".format(
                sorted(mg.track_ids)))
        else:
            print("[INFO] Separation, no crossing — preserved: {}".format(
                sorted(mg.track_ids)))

    def _apply_separation_multi(self, mg: MergeGroup) -> None:
        """
        FIX (h): Apply the permutation mapping for 3+ robot merges.

        entry_order[k] is the track ID that was at position k along entry_axis
        when the merge started.  current_order[k] is the track ID whose peak
        is currently at position k along entry_axis.

        Interpretation: the robot now at slot k (current_order[k]) should
        inherit the identity of the robot that *entered* slot k (entry_order[k]).
        In other words, track entry_order[k] should get the state of current_order[k].

        Example:
          entry_order   = [0, 1, 2]   (left to right at merge start)
          current_order = [2, 0, 1]   (robots rearranged: R2 is now leftmost)
          → R0 should get R2's current peak (slot 0 belongs to entry R0)
          → R1 should get R0's current peak
          → R2 should get R1's current peak
        """
        entry   = mg.entry_order
        current = mg.current_order
        n       = len(entry)

        if len(current) != n:
            # current_order was never fully populated (e.g. peaks always < n)
            # Fall back to simple re-anchor without permutation
            print("[WARN] Multi-robot merge resolved with incomplete order info — "
                  "re-anchoring in place: {}".format(sorted(mg.track_ids)))
            self._reanchor_from_peaks(mg.track_ids, mg.peak_assignment)
            return

        # Build permutation: for each slot k, who's there now vs who should be
        # Slot k should have entry_order[k].  It currently has current_order[k].
        # So entry_order[k] ← state of current_order[k].

        # Snapshot current state for all tracks in the group
        snap_pos    = {tid: self._pos[tid]    for tid in mg.track_ids}
        snap_pos_px = {tid: self._pos_px[tid] for tid in mg.track_ids}
        snap_vel    = {tid: self._vel[tid]    for tid in mg.track_ids}
        snap_vel_px = {tid: self._vel_px[tid] for tid in mg.track_ids}

        swapped = []
        for k in range(n):
            dest_tid = entry[k]    # the ID that owns slot k
            src_tid  = current[k]  # who is physically at slot k right now
            if dest_tid == src_tid:
                continue
            # Assign src's current state to dest
            self._pos[dest_tid]    = snap_pos.get(src_tid)
            self._pos_px[dest_tid] = snap_pos_px.get(src_tid)
            self._vel[dest_tid]    = snap_vel.get(src_tid, (0.0, 0.0))
            self._vel_px[dest_tid] = snap_vel_px.get(src_tid, (0.0, 0.0))
            swapped.append((dest_tid, src_tid))

        # Now re-anchor each track to its peak for a clean separation position
        # (peak_assignment is keyed by track ID after the permutation above
        #  already updated the positions, so we re-anchor by dest_tid → peak
        #  of the physical robot now assigned to that ID)
        for k in range(n):
            dest_tid = entry[k]
            src_tid  = current[k]
            if src_tid in mg.peak_assignment:
                px, py = mg.peak_assignment[src_tid]
                fx, fy = self._image_to_field(px, py)
                if 0 <= fx <= 144 and 0 <= fy <= 144:
                    self._pos[dest_tid]    = (fx, fy)
                    self._pos_px[dest_tid] = (px, py)
                    self._vel[dest_tid]    = (0.0, 0.0)
                    self._vel_px[dest_tid] = (0.0, 0.0)

        if swapped:
            print("[INFO] Multi-robot merge resolved WITH permutation: "
                  "{} — swaps: {}".format(
                      sorted(mg.track_ids),
                      ", ".join("R{}←R{}".format(d, s) for d, s in swapped)))
        else:
            print("[INFO] Multi-robot merge resolved, no permutation: {}".format(
                sorted(mg.track_ids)))

    def _relabel_multi_merge_exit(self, mg: MergeGroup, blobs, assignment: Dict[int, int]) -> bool:
        if not blobs or len(mg.track_ids) <= 2:
            return False

        expected_order = self._resolve_merge_order(mg)
        if len(expected_order) != len(mg.track_ids):
            expected_order = list(mg.current_order or mg.entry_order)
        if len(expected_order) != len(mg.track_ids):
            return False

        slot_for_tid = {tid: slot for slot, tid in enumerate(expected_order)}
        ax, ay = mg.entry_axis
        peak_points = [mg.peak_assignment.get(tid) for tid in mg.track_ids if tid in mg.peak_assignment]
        if not peak_points:
            return False
        center_x = sum(px for px, _py in peak_points) / len(peak_points)
        center_y = sum(py for _px, py in peak_points) / len(peak_points)
        merge_px = self._robot_px_at(center_x, center_y)

        def _proj(px, py):
            return (px - center_x) * ax + (py - center_y) * ay

        candidate_set = set()
        for tid in mg.track_ids:
            if tid in assignment:
                candidate_set.add(assignment[tid])

        blob_ranked = []
        for bi, blob in enumerate(blobs):
            cx, cy = float(blob[4]), float(blob[5])
            nearest_peak = min(
                math.hypot(cx - px, cy - py)
                for px, py in peak_points
            )
            if nearest_peak <= merge_px * 2.6:
                blob_ranked.append((nearest_peak, bi))

        blob_ranked.sort()
        for _dist, bi in blob_ranked:
            candidate_set.add(bi)
            if len(candidate_set) >= min(len(blobs), len(mg.track_ids) + 4):
                break

        # Pull in the top continuation candidates for each track separately.
        # This matters when a robot explodes outward on the split frame and is
        # slightly farther from the merge peaks than the tighter cluster blobs.
        for tid in mg.track_ids:
            peak = mg.peak_assignment.get(tid)
            pred_field = self._predict_field_pos(tid)
            entry_feature = mg.entry_features.get(tid, self._track_features[tid])
            per_track = []
            for bi, blob in enumerate(blobs):
                d_peak = 0.0
                if peak is not None:
                    d_peak = math.hypot(float(blob[4]) - peak[0], float(blob[5]) - peak[1])
                d_field = math.hypot(float(blob[0]) - pred_field[0], float(blob[1]) - pred_field[1])
                appearance_cost = self._appearance_cost(tid, blob)
                entry_feature_cost = self._feature_distance(entry_feature, blob[9] if len(blob) > 9 else None)
                per_track.append((
                    d_peak / max(merge_px, 1.0)
                    + 0.6 * d_field / max(self.MAX_REACQ_IN, 1.0)
                    + 1.4 * appearance_cost
                    + 1.1 * entry_feature_cost,
                    bi,
                ))
            per_track.sort()
            for _score, bi in per_track[:3]:
                candidate_set.add(bi)

        candidate_indices = list(candidate_set)
        if len(candidate_indices) < len(mg.track_ids):
            return False

        sorted_candidates = sorted(candidate_indices, key=lambda bi: _proj(blobs[bi][4], blobs[bi][5]))

        best_cost = float("inf")
        best_map = None
        for chosen in itertools.combinations(sorted_candidates, len(mg.track_ids)):
            total_cost = 0.0
            valid = True
            for slot, (tid, bi) in enumerate(zip(expected_order, chosen)):
                blob = blobs[bi]
                peak = mg.peak_assignment.get(tid)
                if peak is None:
                    peak = self._predict_image_pos(tid)
                pred_field = self._predict_field_pos(tid)
                d_img = math.hypot(float(blob[4]) - peak[0], float(blob[5]) - peak[1])
                d_field = math.hypot(float(blob[0]) - pred_field[0], float(blob[1]) - pred_field[1])
                appearance_cost = self._appearance_cost(tid, blob)
                entry_feature = mg.entry_features.get(tid, self._track_features[tid])
                entry_feature_cost = self._feature_distance(
                    entry_feature,
                    blob[9] if len(blob) > 9 else None,
                )
                slot_penalty = abs(slot - slot_for_tid[tid])
                lock = self._post_merge_locks[tid]
                if lock is not None:
                    lock_img_x, lock_img_y = lock["img_pos"]
                    d_lock = math.hypot(float(blob[4]) - lock_img_x, float(blob[5]) - lock_img_y)
                else:
                    d_lock = 0.0

                total_cost += (
                    d_img / max(merge_px, 1.0)
                    + 0.55 * d_field / max(self.MAX_DIST_IN, 1.0)
                    + 1.9 * appearance_cost
                    + 1.35 * entry_feature_cost
                    + 0.65 * slot_penalty
                    + 0.8 * d_lock / max(merge_px, 1.0)
                )

                if d_img > merge_px * 3.6 and appearance_cost > 0.25:
                    valid = False
                    break

            if valid and total_cost < best_cost:
                best_cost = total_cost
                best_map = {
                    tid: bi
                    for tid, bi in zip(expected_order, chosen)
                }

        if best_map is None:
            return False

        chosen_blob_indices = set(best_map.values())
        for tid, bi in list(assignment.items()):
            if tid not in mg.track_ids and bi in chosen_blob_indices:
                del assignment[tid]
        for tid, bi in best_map.items():
            assignment[tid] = bi
            fx, fy = float(blobs[bi][0]), float(blobs[bi][1])
            cx, cy = float(blobs[bi][4]), float(blobs[bi][5])
            feature = blobs[bi][9] if len(blobs[bi]) > 9 else None
            self._track_features[tid] = feature
            self._prime_post_merge_lock(
                tid,
                (fx, fy),
                (cx, cy),
                prev_image_pos=mg.peak_assignment.get(tid),
                feature=feature,
                merge_group_key=tuple(sorted(mg.track_ids)),
            )

        swaps = []
        for slot, tid in enumerate(expected_order):
            previous_tid = mg.entry_order[slot] if slot < len(mg.entry_order) else tid
            if previous_tid != tid:
                swaps.append("R{}←slot{}".format(previous_tid, slot))
        print("[INFO] Multi-robot merge relabeled from exit blobs: {}".format(
            sorted(mg.track_ids)))
        return True

    def _reanchor_from_peaks(self, track_ids, peak_assignment):
        """Fallback: re-anchor each track to its last-known peak, in-place."""
        reanchored = 0
        for tid in track_ids:
            if tid in peak_assignment:
                px, py = peak_assignment[tid]
                fx, fy = self._image_to_field(px, py)
                if 0 <= fx <= 144 and 0 <= fy <= 144:
                    self._pos[tid]    = (fx, fy)
                    self._pos_px[tid] = (px, py)
                    self._vel[tid]    = (0.0, 0.0)
                    self._vel_px[tid] = (0.0, 0.0)
                    reanchored += 1
        print("[INFO] Re-anchored {}/{} tracks (fallback)".format(
            reanchored, len(track_ids)))

    # ── foreground mask ───────────────────────────────────────────────────

    def _foreground_mask(self, frame):
        cv2, np = self.cv2, self.np
        bg, field_mask = self._background_and_mask()
        diff = cv2.absdiff(frame, bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, self.FG_THRESH, 255, cv2.THRESH_BINARY)
        fg = cv2.bitwise_and(fg, field_mask)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kern)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._kern)

        if self._neck_kern is not None:
            fg = cv2.erode(fg,  self._neck_kern)
            fg = cv2.dilate(fg, self._neck_kern)

        ball_mask = self._ball_color_mask(frame, gray)
        if ball_mask is not None:
            fg = cv2.bitwise_and(fg, cv2.bitwise_not(ball_mask))

        return fg

    def _scaled_odd_kernel(self, frac: float, min_size: int):
        cv2 = self.cv2
        size = max(min_size, int(round(self._robot_max_px * frac)))
        if size % 2 == 0:
            size += 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _ball_color_mask(self, frame, diff_gray=None):
        """
        Builds two masks from the scoring-element colors (season.py):

        - returned / _last_ball_suppress_mask: ball-colored pixels on the
          field (moving or not), cleaned and dilated; removed from the robot
          foreground so game pieces are not mistaken for robots.
        - _last_ball_mask: ball-colored pixels that also differ from the
          background (i.e. moving or newly placed), inside the shot region
          (field + space above the far wall around the scoring targets).
          Used by the shot detector. Only lightly cleaned so small, distant
          balls survive.
        """
        if not self.BALL_FILTER_ENABLED or self._field_mask is None:
            self._last_ball_mask = None
            self._last_ball_suppress_mask = None
            return None

        cv2, np = self.cv2, self.np
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color = None
        for lo, hi in self.BALL_HSV_RANGES:
            part = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            color = part if color is None else cv2.bitwise_or(color, part)
        if color is None:
            self._last_ball_mask = None
            self._last_ball_suppress_mask = None
            return None

        # Moving-ball mask for shot detection.
        if diff_gray is None:
            diff_gray = cv2.cvtColor(cv2.absdiff(frame, self._bg), cv2.COLOR_BGR2GRAY)
        _, moving = cv2.threshold(diff_gray, self.BALL_MOTION_THRESH, 255, cv2.THRESH_BINARY)
        shot = cv2.bitwise_and(color, moving)
        region = self._shot_region_mask if self._shot_region_mask is not None else self._field_mask
        shot = cv2.bitwise_and(shot, region)
        if self.BALL_SHOT_CLOSE_PX and self.BALL_SHOT_CLOSE_PX > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.BALL_SHOT_CLOSE_PX, self.BALL_SHOT_CLOSE_PX))
            shot = cv2.morphologyEx(shot, cv2.MORPH_CLOSE, k)
        self._last_ball_mask = shot

        # Suppression mask for robot segmentation.
        mask = cv2.bitwise_and(color, self._field_mask)

        open_kern = self._scaled_odd_kernel(self.BALL_MASK_OPEN_FRAC, 3)
        close_kern = self._scaled_odd_kernel(self.BALL_MASK_CLOSE_FRAC, 3)
        dilate_kern = self._scaled_odd_kernel(self.BALL_MASK_DILATE_FRAC, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kern)
        mask = self._filter_contours_by_field_area(mask, self.BALL_MIN_FIELD_AREA_IN2)
        mask = cv2.dilate(mask, dilate_kern)
        self._last_ball_suppress_mask = mask
        return mask

    def _filter_contours_by_field_area(self, mask, min_area_in2: float):
        cv2, np = self.cv2, self.np
        if self._H_2d is None or min_area_in2 <= 0:
            return mask

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = np.zeros_like(mask)
        for c in cnts:
            if len(c) < 3:
                continue
            if self._contour_field_area_in2(c) >= min_area_in2:
                cv2.drawContours(kept, [c], -1, 255, -1)
        return kept

    def _contour_field_area_in2(self, contour) -> float:
        cv2, np = self.cv2, self.np
        if self._H_2d is None:
            return 0.0
        approx = cv2.approxPolyDP(contour, 1.5, True)
        if len(approx) < 3:
            return 0.0
        pts = approx.reshape(-1, 2).astype(np.float32)
        field = cv2.perspectiveTransform(np.array([pts], dtype=np.float32),
                                         self._H_2d)[0]
        return abs(float(cv2.contourArea(field.reshape(-1, 1, 2))))

    # ── blob detector ─────────────────────────────────────────────────────

    def _foreground_contours(self, frame):
        cv2, np = self.cv2, self.np
        fg = self._foreground_mask(frame)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in cnts:
            if cv2.contourArea(c) < self._blob_min_area(c):
                continue
            if self._contour_field_area_in2(c) < self.BLOB_MIN_FIELD_AREA_IN2:
                continue
            valid.append(c)
        valid.sort(key=lambda c: -cv2.contourArea(c))
        all_split = []
        for c in valid:
            area = float(cv2.contourArea(c))
            subs = self._split_contour(c)
            per  = area / max(len(subs), 1)
            for sx, sy in subs:
                all_split.append((sx, sy, per))
        all_split.sort(key=lambda x: -x[2])
        return valid, [(int(sx), int(sy)) for sx, sy, _ in all_split]

    def _split_contour(self, contour):
        cv2, np = self.cv2, self.np
        h, w = self._field_mask.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return []
        _, peak_mask = cv2.threshold(dist, dist.max() * 0.40, 255, cv2.THRESH_BINARY)
        peak_mask = peak_mask.astype(np.uint8)
        k_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        peak_mask = cv2.erode(peak_mask, k_sep)
        n_labels, labels = cv2.connectedComponents(peak_mask)
        centers = []
        for lbl in range(1, n_labels):
            comp = (labels == lbl).astype(np.uint8)
            M = cv2.moments(comp)
            if M["m00"] > 0:
                centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        area = float(cv2.contourArea(contour))
        expected = self._expected_robot_count(area, contour)
        if len(centers) < expected:
            retry = self._split_contour_relaxed(mask, dist, expected)
            if len(retry) > len(centers):
                centers = retry
        return centers

    def _expected_robot_count(self, contour_area: float, contour=None) -> int:
        robot_px = self._contour_robot_px(contour)
        robot_area = max(robot_px * robot_px * self._single_area_ratio, 1.0)
        ratio = contour_area / robot_area
        if ratio >= self.EXPECT_4WAY_AREA:
            return 4
        if ratio >= self.EXPECT_3WAY_AREA:
            return 3
        if ratio >= self.EXPECT_2WAY_AREA:
            return 2
        return 1

    def _split_contour_relaxed(self, mask, dist, expected: int):
        cv2, np = self.cv2, self.np
        if dist.max() == 0:
            return []
        peak_ratio = (self.RELAXED_PEAK_RATIO_MULTI
                      if expected >= 3 else self.RELAXED_PEAK_RATIO_PAIR)
        _, peak_mask = cv2.threshold(dist, dist.max() * peak_ratio, 255, cv2.THRESH_BINARY)
        peak_mask = peak_mask.astype(np.uint8)
        sep_size = self.RELAXED_SEP_MULTI if expected >= 3 else self.RELAXED_SEP_PAIR
        k_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sep_size, sep_size))
        peak_mask = cv2.erode(peak_mask, k_sep)
        peak_mask = cv2.bitwise_and(peak_mask, mask)
        n_labels, labels = cv2.connectedComponents(peak_mask)
        centers = []
        for lbl in range(1, n_labels):
            comp = (labels == lbl).astype(np.uint8)
            M = cv2.moments(comp)
            if M["m00"] > 0:
                centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        return centers

    def _find_underresolved_tracks(self, blobs: list):
        blocked = set()
        seen = {}
        for blob in blobs:
            pid = blob[3]
            if pid in seen:
                continue
            seen[pid] = (blob[8], blob[7])

        for contour, split_count in seen.values():
            if contour is None:
                continue
            expected = self._expected_robot_count(float(self.cv2.contourArea(contour)), contour)
            nearby_tracks = []
            for track_id in range(4):
                if self._pos[track_id] is None and self._pos_px[track_id] is None:
                    continue
                if season.LOST_TRACK_RULES and self._coast[track_id] > self.VISIBLE_COAST:
                    continue
                pred_px_x, pred_px_y = self._predict_image_pos(track_id)
                dist = self.cv2.pointPolygonTest(
                    contour, (float(pred_px_x), float(pred_px_y)), True)
                if dist >= -self._track_robot_px(track_id) * 0.75:
                    nearby_tracks.append(track_id)
            expected = max(expected, len(nearby_tracks))
            if expected <= split_count:
                continue
            blocked.update(nearby_tracks)
        return blocked

    def _get_blobs(self, frame):
        cv2, np = self.cv2, self.np
        self._frame_counter += 1
        fg = self._foreground_mask(frame)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for parent_id, c in enumerate(cnts):
            area = float(cv2.contourArea(c))
            if area < self._blob_min_area(c):
                continue
            field_area = self._contour_field_area_in2(c)
            if field_area < self.BLOB_MIN_FIELD_AREA_IN2:
                continue
            _bm = np.zeros(self._field_mask.shape, dtype=np.uint8)
            cv2.drawContours(_bm, [c], -1, 255, -1)
            _dr = cv2.distanceTransform(_bm, cv2.DIST_L2, 5)
            if _dr.max() < self.MIN_RADIUS_PX:
                continue
            sub_centers = self._split_contour(c)
            sub_centers = self._augment_subcenters_with_track_priors(c, sub_centers)
            if not sub_centers:
                continue
            per_area = area / len(sub_centers)
            quality_penalty = self._blob_quality_penalty(
                area, per_area, len(sub_centers), self._contour_robot_px(c))
            contour_bottom = float(c[:, 0, 1].max())
            for cx, cy in sub_centers:
                fx, fy = self._image_to_field(cx, cy, bottom_v=contour_bottom)
                if not (0 <= fx <= 144 and 0 <= fy <= 144):
                    # With the floor-contact correction, robots touching a
                    # wall can project slightly outside; keep them on the edge.
                    tol = self.FIELD_EDGE_TOL_IN
                    if (season.FOOT_OFFSET_FRAC > 0
                            and -tol <= fx <= 144 + tol and -tol <= fy <= 144 + tol):
                        fx = min(max(fx, 0.0), 144.0)
                        fy = min(max(fy, 0.0), 144.0)
                    else:
                        continue
                # ── Corner zone suppression ─────────────────────────────────
                # Blobs inside the top-left or top-right corner zones are
                # suppressed unless a robot is known to be actively moving there.
                cz = self.CORNER_ZONE_IN
                in_corner_zone = (
                    (fx < cz and fy < cz) or           # top-left corner
                    (fx > 144 - cz and fy < cz)        # top-right corner
                )
                if season.CORNER_ZONE_SUPPRESSION and self._initialized and in_corner_zone:
                    # Allow if an actively tracked robot is already nearby.
                    robot_present = False
                    for tid in range(4):
                        if self._pos[tid] is None:
                            continue
                        tx, ty = self._pos[tid]
                        dist = math.hypot(fx - tx, fy - ty)
                        if dist < self.ROBOT_SIZE_IN * 1.5 and self._coast[tid] <= self.VISIBLE_COAST:
                            robot_present = True
                            break
                    if not robot_present:
                        continue
                # ── end corner zone suppression ─────────────────────────────
                appearance = self._extract_appearance_feature(frame, int(cx), int(cy), c)
                candidates.append((fx, fy, 0.0, per_area, parent_id,
                                    int(cx), int(cy), quality_penalty,
                                    len(sub_centers), c, appearance))

        candidates.sort(key=lambda b: -b[3])

        # ── Static blob suppression ─────────────────────────────────────────
        # Corner structures and other stationary FG objects create persistent
        # blobs that confuse robot tracking.  Any candidate whose centroid has
        # stayed within STATIC_BLOB_MOVE_PX pixels for STATIC_BLOB_SUPPRESS_FRAMES
        # consecutive frames is dropped UNLESS a current track is already assigned
        # very close to it (which would mean it really is a robot sitting still).
        if self._initialized:
            filtered = []
            new_static: dict = {}
            new_static_positions: dict = {}
            for entry in candidates:
                fx, fy, hdg, _a, pid, cx, cy, qual, nsplit, cnt, appearance = entry
                # Bucket this centroid to a coarse grid for history lookup
                key = (round(cx / self.STATIC_BLOB_MOVE_PX),
                       round(cy / self.STATIC_BLOB_MOVE_PX))
                # Find closest existing static key
                matched_key = None
                for k in self._blob_static_positions:
                    okx, oky = self._blob_static_positions[k]
                    if abs(cx - okx) < self.STATIC_BLOB_MOVE_PX and abs(cy - oky) < self.STATIC_BLOB_MOVE_PX:
                        matched_key = k
                        break
                if matched_key is not None:
                    count = self._blob_static_counts.get(matched_key, 0) + 1
                    new_static[matched_key] = count
                    new_static_positions[matched_key] = (cx, cy)
                    near_track = any(
                        self._pos[tid] is not None
                        and math.hypot(fx - self._pos[tid][0], fy - self._pos[tid][1]) < self.ROBOT_SIZE_IN * 1.5
                        and (self._coast[tid] <= self.VISIBLE_COAST
                             or season.STATIC_SUPPRESS_SPARES_LOST_TRACKS)
                        for tid in range(4)
                    )
                    if count >= self.STATIC_BLOB_SUPPRESS_FRAMES and not near_track:
                        continue
                else:
                    new_static[key] = 1
                    new_static_positions[key] = (cx, cy)
                filtered.append(entry)
            self._blob_static_counts = new_static
            self._blob_static_positions = new_static_positions
            candidates = filtered
        else:
            self._blob_static_counts = {}
            self._blob_static_positions = {}
        # ── end static blob suppression ─────────────────────────────────────

        return [(fx, fy, hdg, pid, cx, cy, qual, nsplit, cnt, appearance)
                for fx, fy, hdg, _a, pid, cx, cy, qual, nsplit, cnt, appearance in candidates]

    def _augment_subcenters_with_track_priors(self, contour, centers):
        expected = self._expected_robot_count(float(self.cv2.contourArea(contour)), contour)
        if len(centers) >= expected:
            return centers
        robot_px = self._contour_robot_px(contour)

        nearby = []
        for track_id in range(4):
            if self._pos[track_id] is None and self._pos_px[track_id] is None:
                continue
            if season.LOST_TRACK_RULES and self._coast[track_id] > self.VISIBLE_COAST:
                continue
            px, py = self._predict_image_pos(track_id)
            dist = self.cv2.pointPolygonTest(contour, (float(px), float(py)), True)
            if dist >= -robot_px * self.MERGE_PRIOR_PAD_PX:
                nearby.append((dist, float(px), float(py)))

        target = min(max(expected, len(nearby)), 4)
        if len(centers) >= target or not nearby:
            return centers

        augmented = list(centers)
        min_sep = max(robot_px * self.MERGE_PRIOR_MIN_SEP, 6.0)
        nearby.sort(reverse=True)
        for _dist, px, py in nearby:
            too_close = any(math.hypot(px-cx, py-cy) < min_sep for cx, cy in augmented)
            if too_close:
                continue
            augmented.append((px, py))
            if len(augmented) >= target:
                break
        return augmented

    def _blob_quality_penalty(self, area: float, per_area: float, split_count: int,
                              robot_px: Optional[float] = None) -> float:
        robot_px = self._robot_max_px if robot_px is None else robot_px
        robot_area = max(robot_px * robot_px * self._single_area_ratio, 1.0)
        ratio = per_area / robot_area
        penalty = min(abs(math.log(max(ratio, 1e-6))), 2.5) * 0.28
        if split_count == 1 and area > robot_area * 1.8:
            penalty += min((area / robot_area) - 1.8, 2.0) * 0.7
        return penalty

    def _extract_appearance_feature(self, frame, cx: int, cy: int, contour=None):
        cv2, np = self.cv2, self.np
        if frame is None:
            return None

        radius = max(6, int(round(self._robot_px_at(cx, cy) * self.REID_CROP_SCALE)))
        x0 = max(0, int(cx) - radius)
        y0 = max(0, int(cy) - radius)
        x1 = min(frame.shape[1], int(cx) + radius + 1)
        y1 = min(frame.shape[0], int(cy) + radius + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None

        crop = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.circle(mask, (int(cx) - x0, int(cy) - y0), max(3, radius - 1), 255, -1)
        if contour is not None:
            shifted = contour.copy()
            shifted[:, 0, 0] -= x0
            shifted[:, 0, 1] -= y0
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [shifted], -1, 255, -1)
            mask = cv2.bitwise_and(mask, contour_mask)

        hist = cv2.calcHist(
            [hsv], [0, 1], mask,
            [self.REID_HIST_H_BINS, self.REID_HIST_S_BINS],
            [0, 180, 0, 256],
        )
        if hist is None:
            return None
        hist = hist.astype(np.float32)
        total = float(hist.sum())
        if total <= 0.0:
            return None
        hist /= total
        return hist

    def _appearance_cost(self, track_id: int, blob) -> float:
        if len(blob) < 10 or blob[9] is None:
            return 0.0
        return self._appearance_cost_for_feature(track_id, blob[9])

    def _feature_distance(self, feature_a, feature_b) -> float:
        if feature_a is None or feature_b is None:
            return 0.0
        return float(self.cv2.compareHist(feature_a, feature_b, self.cv2.HISTCMP_BHATTACHARYYA))

    def _auto_refs(self, track_id: int) -> List:
        return self._auto_refs_early[track_id] + self._auto_refs_recent[track_id]

    def _isolated_assignment(self, track_id: int, blobs, assignment: Dict[int, int]) -> bool:
        if track_id not in assignment:
            return False
        blob = blobs[assignment[track_id]]
        if blob[7] != 1 or (len(blob) > 9 and blob[9] is None):
            return False
        if any(track_id in mg.track_ids for mg in self._merge_groups.values()):
            return False
        if any(t != track_id and blobs[a][3] == blob[3] for t, a in assignment.items()):
            return False
        for t in range(4):
            if t == track_id or self._pos[t] is None or self._coast[t] > self.VISIBLE_COAST:
                continue
            if math.hypot(blob[0] - self._pos[t][0], blob[1] - self._pos[t][1]) < self.AUTO_REID_ISOLATION_IN:
                return False
        return True

    def _early_cost(self, identity: int, feature) -> Optional[float]:
        bank = self._auto_refs_early[identity]
        if feature is None or len(bank) < self.IDFIX_MIN_EARLY:
            return None
        dists = sorted(self._feature_distance(feature, ref) for ref in bank)
        k = min(3, len(dists))
        return sum(dists[:k]) / k

    def _update_identity_evidence(self, blobs, assignment: Dict[int, int]) -> None:
        if not (season.AUTO_REID and season.APPEARANCE_ID_CORRECTION) or self._reid_refs:
            return
        np = self.np
        trusted = [k for k in range(4) if len(self._auto_refs_early[k]) >= self.IDFIX_MIN_EARLY]
        if len(trusted) < 2:
            return
        for i in range(4):
            if not self._isolated_assignment(i, blobs, assignment):
                continue
            feature = blobs[assignment[i]][9]
            costs = np.full(4, np.nan)
            for k in trusted:
                costs[k] = self._early_cost(k, feature)
            if self._app_ema[i] is None:
                self._app_ema[i] = costs
            else:
                prev = self._app_ema[i]
                upd = np.where(np.isnan(prev), costs, prev * (1 - self.IDFIX_EMA) + costs * self.IDFIX_EMA)
                self._app_ema[i] = np.where(np.isnan(costs), prev, upd)
            self._app_obs[i] += 1
            self._app_last_t[i] = self._now
        self._maybe_correct_identities(trusted)

    def _maybe_correct_identities(self, trusted) -> None:
        np = self.np
        rows = [
            i for i in range(4)
            if self._app_ema[i] is not None and self._app_obs[i] >= self.IDFIX_MIN_OBS
            and self.tracked_poses[i].visible and self._coast[i] <= self.VISIBLE_COAST
            and self._app_last_t[i] is not None and self._now is not None
            and self._now - self._app_last_t[i] <= self.IDFIX_MAX_AGE_S
            and not any(i in mg.track_ids for mg in self._merge_groups.values())
            and i in trusted
        ]
        if len(rows) < 1:
            self._idfix_votes = 0
            return
        cols = list(trusted)
        cost = np.array([[self._app_ema[i][k] if not np.isnan(self._app_ema[i][k]) else 1.0
                          for k in cols] for i in rows])
        try:
            from scipy.optimize import linear_sum_assignment
            r_ind, c_ind = linear_sum_assignment(cost)
        except ImportError:
            self._idfix_votes = 0
            return
        mapping = {rows[r]: cols[c] for r, c in zip(r_ind, c_ind)}
        changed = {i: k for i, k in mapping.items() if i != k}
        if not changed:
            self._idfix_votes = 0
            self._idfix_key = None
            return
        # Only move a label onto an identity that is either among the rows
        # (visible, being relabeled too) or currently not visible.
        for i, k in changed.items():
            if k not in rows and self.tracked_poses[k].visible:
                self._idfix_votes = 0
                self._idfix_key = None
                return
        cur = sum(cost[rows.index(i)][cols.index(i)] for i in changed)
        new = sum(cost[rows.index(i)][cols.index(k)] for i, k in changed.items())
        if cur - new < self.IDFIX_MARGIN * len(changed):
            self._idfix_votes = 0
            self._idfix_key = None
            return
        key = tuple(sorted(changed.items()))
        if key == self._idfix_key:
            self._idfix_votes += 1
        else:
            self._idfix_key = key
            self._idfix_votes = 1
        if self._idfix_votes < self.IDFIX_VOTES:
            return
        self._apply_identity_permutation(changed)
        self._idfix_votes = 0
        self._idfix_key = None

    def _apply_identity_permutation(self, changed: Dict[int, int]) -> None:
        """changed: {current track id: identity it really is}. Track state
        moves to the true identity; identities left without a robot get the
        leftover states (usually hidden)."""
        sources = list(changed.keys())
        targets = list(changed.values())
        leftover_targets = [s for s in sources if s not in targets]
        leftover_sources = [t for t in targets if t not in sources]
        perm = dict(changed)  # source -> target
        for s_id, t_id in zip(leftover_sources, leftover_targets):
            perm[s_id] = t_id
        fields = ["_pos", "_pos_px", "_vel", "_vel_px", "_coast", "_merge_recent",
                  "_post_merge_locks", "_track_features", "_last_seen_t",
                  "_app_ema", "_app_obs", "_app_last_t"]
        snap = {f: list(getattr(self, f)) for f in fields}
        pose_snap = [(p.x_in, p.y_in, p.heading, p.visible) for p in self.tracked_poses]
        for src, dst in perm.items():
            for f in fields:
                getattr(self, f)[dst] = snap[f][src]
            pose = self.tracked_poses[dst]
            pose.x_in, pose.y_in, pose.heading, pose.visible = pose_snap[src]
        for ident in perm.values():
            # recent samples were taken while this label may have been wrong
            self._auto_refs_recent[ident] = []
        print("[INFO] Identity correction by appearance: {}".format(
            ", ".join("robot on R{} is R{}".format(s, d) for s, d in sorted(changed.items()))))

    def _maybe_add_auto_ref(self, track_id: int, blobs, assignment: Dict[int, int]) -> None:
        """Store this track's appearance while it is clearly on its own."""
        if not season.AUTO_REID or self._reid_refs:
            return
        blob = blobs[assignment[track_id]]
        feature = blob[9] if len(blob) > 9 else None
        if feature is None or blob[7] != 1:
            return
        if any(track_id in mg.track_ids for mg in self._merge_groups.values()):
            return
        if any(t != track_id and blobs[a][3] == blob[3] for t, a in assignment.items()):
            return
        fx, fy = blob[0], blob[1]
        for t in range(4):
            if t == track_id or self._pos[t] is None or self._coast[t] > self.VISIBLE_COAST:
                continue
            if math.hypot(fx - self._pos[t][0], fy - self._pos[t][1]) < self.AUTO_REID_ISOLATION_IN:
                return
        self._auto_ref_counter[track_id] += 1
        if self._auto_ref_counter[track_id] % self.AUTO_REID_STRIDE != 0:
            return
        bank = self._auto_refs(track_id)
        if len(bank) >= 8:
            nearest = min(self._feature_distance(feature, ref) for ref in bank)
            if nearest > self.AUTO_REID_ADD_MAX_DIST:
                return   # looks unlike this robot: maybe the track is on the wrong one
        early = self._auto_refs_early[track_id]
        if len(early) < self.AUTO_REID_EARLY_SAMPLES:
            early.append(feature)
        else:
            recent = self._auto_refs_recent[track_id]
            recent.append(feature)
            if len(recent) > self.AUTO_REID_RECENT_SAMPLES:
                recent.pop(0)

    def _appearance_cost_for_feature(self, track_id: int, feature) -> float:
        if feature is None:
            return 0.0
        if not self._reid_refs or track_id not in self._reid_refs:
            if not season.AUTO_REID:
                return 0.0
            bank = self._auto_refs(track_id)
            if len(bank) < self.AUTO_REID_MIN_SAMPLES:
                return 0.0
            dists = sorted(self._feature_distance(feature, ref) for ref in bank)
            k = min(3, len(dists))
            return sum(dists[:k]) / k

        refs = self._reid_refs.get(track_id) or []
        if not refs:
            return 0.0

        cv2 = self.cv2
        best = 1.0
        for ref in refs:
            d = float(cv2.compareHist(feature, ref, cv2.HISTCMP_BHATTACHARYYA))
            if d < best:
                best = d
        return best

    def set_reid_reference_histograms(self, refs: Dict[int, List]) -> None:
        self._reid_refs = refs if refs else None
