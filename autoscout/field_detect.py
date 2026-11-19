"""
Automatic field calibration from match video.

Finds the four field corners at FLOOR level, even when some corners are
outside the frame or hidden under a broadcast overlay, plus the HIVE boxes,
FLOWER boxes and overlay exclusion bands. The result has the same format as
field_corners.json and is meant to be reviewed (and adjusted if needed) by a
person before a match is processed.

How it works
1. A median background of frames sampled across the video removes robots,
   balls and most people.
2. The field carpet is segmented by color from a seed near the image center
   (light, low-saturation), then cleaned up.
3. Straight lines are fitted (RANSAC) to the carpet boundary: the back floor
   line and the left / right floor lines (where the carpet meets the inside
   bottom of the walls). Boundary points on the image border or on an overlay
   are ignored, so the lines can be extrapolated to corners that are not
   visible.
4. The front edge: if the carpet boundary along the front is visible, that is
   the front floor line. Usually it is not (the front wall's top rail crosses
   the view, and the bottom of the wall is below the frame or under the
   overlay). Then the front TOP rail is found as a dark line across the lower
   carpet and the floor line is placed below it by the wall's projected
   height. The wall height in pixels is self-calibrated at the back wall, where
   both the floor edge and the top rail are visible, using a local
   weak-perspective model (vertical px per inch = lateral px per inch x
   cos(depression angle), with sin(angle) = depth / lateral foreshortening).
5. Corners are the line intersections (in image coordinates, possibly outside
   the frame).
"""
import math
from typing import Dict, List, Optional, Tuple

from autoscout import season

FIELD_IN = 144.0


def _line_from_points(np, pts):
    """Total-least-squares line through points: returns (point, unit dir)."""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - c)
    d = vt[0]
    return c, d / max(np.linalg.norm(d), 1e-12)


def _ransac_line(np, pts, thresh: float, iters: int = 300, rng=None):
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 8:
        return None
    rng = rng or np.random.default_rng(7)
    best_inl = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        p, q = pts[i], pts[j]
        d = q - p
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            continue
        nvec = np.array([-d[1], d[0]]) / norm
        dist = np.abs((pts - p) @ nvec)
        inl = dist < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl is None or best_inl.sum() < 8:
        return None
    c, d = _line_from_points(np, pts[best_inl])
    return {"point": c, "dir": d, "inliers": int(best_inl.sum()), "support": pts[best_inl]}


def _ransac_outer_line(np, pts, thresh: float, region_pts, outward, max_outside: float = 0.02,
                       iters: int = 400, rng=None):
    """RANSAC line through boundary points that has (almost) no region pixels
    on its outer side. `outward` is a unit vector pointing away from the
    region; the true wall line is the outer limit of the carpet, while edges of
    notches (FLOWERS, people at the walls) have carpet beyond them."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 8:
        return None
    rng = rng or np.random.default_rng(11)
    region_pts = np.asarray(region_pts, dtype=np.float64)
    best = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        p, q = pts[i], pts[j]
        d = q - p
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            continue
        nvec = np.array([-d[1], d[0]]) / norm
        if nvec @ outward < 0:
            nvec = -nvec
        dist = (pts - p) @ nvec
        inl = np.abs(dist) < thresh
        cnt = int(inl.sum())
        if best is not None and cnt <= best[0]:
            continue
        outside = float(((region_pts - p) @ nvec > thresh * 1.5).mean())
        if outside > max_outside:
            continue
        best = (cnt, inl)
    if best is None or best[0] < 8:
        return None
    c, d = _line_from_points(np, pts[best[1]])
    return {"point": c, "dir": d, "inliers": int(best[0]), "support": pts[best[1]]}


def _intersect(np, l1, l2) -> Optional[Tuple[float, float]]:
    p, r = l1["point"], l1["dir"]
    q, s = l2["point"], l2["dir"]
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-9:
        return None
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / cross
    x = p + t * r
    return float(x[0]), float(x[1])


def _y_on_line(line, x: float) -> float:
    p, d = line["point"], line["dir"]
    if abs(d[0]) < 1e-9:
        return float(p[1])
    t = (x - p[0]) / d[0]
    return float(p[1] + t * d[1])



# How far outside the camera view a field corner may be, as a fraction of the
# larger frame side. Matches the dashboard calibration editor's maximum border.
CORNER_MAX_OUTSIDE_FRAC = 0.5

class FieldCalibrator:
    N_SAMPLES = 60
    WALL_HEIGHT_IN = 12.0          # FTC perimeter wall height (fallback)

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self.debug: Dict[str, object] = {}

    # ── public ────────────────────────────────────────────────────────────
    def calibrate_video(self, video_path: str, start_s: float = 0.0, end_s: Optional[float] = None):
        cv2, np = self.cv2, self.np
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        lo = int(max(0.0, start_s) * fps)
        hi = n - 1 if end_s is None else min(n - 1, int(end_s * fps))
        frames = []
        for idx in np.linspace(lo, max(lo, hi), self.N_SAMPLES).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
        if not frames:
            return None
        return self.calibrate_frames(frames)

    def calibrate_frames(self, frames) -> Optional[Dict[str, object]]:
        np = self.np
        bg = np.median(np.array(frames), axis=0).astype(np.uint8)
        self.debug = {"background": bg}
        overlay_bands = self._overlay_bands(bg, frames)
        self.debug["overlay_bands"] = overlay_bands
        carpet = self._carpet_mask(bg, overlay_bands, frames)
        if carpet is None:
            return None
        self.debug["carpet"] = carpet
        lines = self._boundary_lines(bg, carpet, overlay_bands)
        if lines is None:
            return None
        corners = self._corners(bg, carpet, lines, overlay_bands)
        if corners is None:
            return None
        tl, tr, br, bl = corners["corners"]
        # Corners may lie outside the view (floor-level corners below a low
        # camera), but never further out than the dashboard editor can show,
        # so they can always be checked and dragged. Further out than that
        # means a bad detection: clamp and mark it for checking.
        fh, fw = bg.shape[:2]
        lim = CORNER_MAX_OUTSIDE_FRAC * max(fw, fh)
        clamped = []
        for p in (tl, tr, br, bl):
            q = (min(max(float(p[0]), -lim), fw + lim), min(max(float(p[1]), -lim), fh + lim))
            clamped.append(q)
        if any(abs(a[0] - b[0]) > 0.5 or abs(a[1] - b[1]) > 0.5 for a, b in zip((tl, tr, br, bl), clamped)):
            corners["confidence"] = "low"
            corners["front_method"] = str(corners.get("front_method", "")) + \
                " | a corner was far outside the view and was pulled in - check it"
        tl, tr, br, bl = clamped
        result = {
            "corners_px": [[round(bl[0], 1), round(bl[1], 1)], [round(br[0], 1), round(br[1], 1)],
                           [round(tr[0], 1), round(tr[1], 1)], [round(tl[0], 1), round(tl[1], 1)]],
            "auto": {
                "method": corners["front_method"],
                "visible_corners": corners["visible"],
                "confidence": corners["confidence"],
                "wall_height_in": corners.get("wall_height_in"),
                "notes": corners.get("notes", []),
            },
        }
        exclude = [[[0, y0], [bg.shape[1] - 1, y0], [bg.shape[1] - 1, y1], [0, y1]]
                   for (y0, y1) in overlay_bands]
        if exclude:
            result["exclude_px"] = exclude
        hives = self._hive_boxes(frames, corners)
        if hives:
            # HIVE boxes must not reach into a broadcast overlay band (its
            # red/blue banners would swamp tip detection)
            h_img = bg.shape[0]
            for color, box in list(hives.items()):
                y0, y1 = box[0][1], box[2][1]
                for b0, b1 in overlay_bands:
                    if b0 <= 0 + 3 and y0 <= b1:        # band at the top
                        y0 = b1 + 1
                    if b1 >= h_img - 4 and y1 >= b0:    # band at the bottom
                        y1 = b0 - 1
                if y1 - y0 < 8:
                    del hives[color]
                    continue
                box[0][1] = box[1][1] = int(y0)
                box[2][1] = box[3][1] = int(y1)
            result["hive_boxes_px"] = hives
        flowers = self._flower_boxes(bg, corners)
        if flowers:
            result["flower_boxes_px"] = flowers
        return result

    # ── overlay ───────────────────────────────────────────────────────────
    def _overlay_bands(self, bg, frames) -> List[Tuple[int, int]]:
        """Broadcast overlay bands touching the top or bottom edge: rows that
        are mostly flat overlay colors (saturated red/blue, white, black)."""
        cv2, np = self.cv2, self.np
        h, w = bg.shape[:2]
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
        flat = ((s > 110) & (v > 90)) | ((v > 225) & (s < 35)) | (v < 22)
        frac = flat.mean(axis=1)
        is_overlay = frac > 0.55
        bands = []
        # bottom band
        y = h - 1
        while y > h * 0.5 and (is_overlay[y] or is_overlay[max(0, y - 3):y].mean() > 0.5):
            y -= 1
        if y < h - 6:
            bands.append((y + 1, h - 1))
        y = 0
        while y < h * 0.5 and (is_overlay[y] or is_overlay[y:y + 4].mean() > 0.5):
            y += 1
        if y > 5:
            bands.append((0, y - 1))
        # pad a few px so anti-aliased overlay edges are covered
        return [(max(0, a - 3), min(h - 1, b + 3)) for a, b in bands]

    # ── carpet ────────────────────────────────────────────────────────────
    def _carpet_mask(self, bg, overlay_bands, frames=None):
        cv2, np = self.cv2, self.np
        h, w = bg.shape[:2]
        lab = cv2.cvtColor(cv2.GaussianBlur(bg, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = lab[:, :, 0], lab[:, :, 1] - 128, lab[:, :, 2] - 128
        chroma = np.hypot(A, B)
        valid = np.ones((h, w), bool)
        for y0, y1 in overlay_bands:
            valid[y0:y1 + 1] = False
        # seed: light, low-chroma pixels in the central region
        ys, xs = np.mgrid[0:h, 0:w]
        central = (np.abs(xs - w / 2) < w * 0.22) & (ys > h * 0.35) & (ys < h * 0.85) & valid
        cand = central & (chroma < 14) & (L > 90)
        if cand.sum() < 200:
            cand = central & (chroma < 18)
            if cand.sum() < 200:
                return None
        Lm = float(np.median(L[cand]))
        Am = float(np.median(A[cand]))
        Bm = float(np.median(B[cand]))
        # carpet: similar chroma, lightness not far below the seed (white tape
        # and glare are allowed above)
        def carpet_like(lab_img):
            Li, Ai, Bi = lab_img[:, :, 0], lab_img[:, :, 1] - 128, lab_img[:, :, 2] - 128
            return (np.hypot(Ai - Am, Bi - Bm) < 12) & (Li > Lm - 32) & (Li < Lm + 70)
        mask = carpet_like(lab) & valid
        if frames:
            # A pixel is carpet if it looks like carpet in enough frames: robots
            # parked against a wall for most of the match (still present in the
            # median background) then stop hiding the floor under them.
            count = np.zeros((h, w), np.int32)
            for f in frames:
                lf = cv2.cvtColor(cv2.GaussianBlur(f, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
                count += carpet_like(lf)
            mask = mask | ((count >= max(2, int(0.2 * len(frames)))) & valid)
        mask = mask.astype(np.uint8) * 255
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
        # bridge thin dark rails / tape lines inside the field
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(5, h // 40))))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n_lab, labels, stats, _c = cv2.connectedComponentsWithStats(mask)
        if n_lab <= 1:
            return None
        # component overlapping the seed area the most
        best, best_score = None, 0
        for k in range(1, n_lab):
            score = int(((labels == k) & cand).sum())
            if score > best_score:
                best, best_score = k, score
        if best is None:
            return None
        carpet = (labels == best).astype(np.uint8) * 255
        # fill holes (robots parked all match, HIVE frame, tape)
        cnts, _ = cv2.findContours(carpet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(carpet)
        cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
        return filled

    # ── boundary lines ────────────────────────────────────────────────────
    def _boundary_lines(self, bg, carpet, overlay_bands):
        cv2, np = self.cv2, self.np
        h, w = carpet.shape[:2]
        cnts, _ = cv2.findContours(carpet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        # drop points on the image border or on an overlay edge
        keep = (cnt[:, 0] > 2) & (cnt[:, 0] < w - 3) & (cnt[:, 1] > 2) & (cnt[:, 1] < h - 3)
        for y0, y1 in overlay_bands:
            keep &= ~((cnt[:, 1] >= y0 - 3) & (cnt[:, 1] <= y1 + 3))
        n = len(cnt)
        # local tangent direction
        k = 6
        prev = cnt[(np.arange(n) - k) % n]
        nxt = cnt[(np.arange(n) + k) % n]
        tang = nxt - prev
        ang = np.degrees(np.arctan2(tang[:, 1], tang[:, 0]))
        ang = (ang + 180.0) % 180.0          # 0..180, 0/180 = horizontal
        horiz = (ang < 20) | (ang > 160)
        M = cv2.moments(carpet)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        ys = cnt[:, 1]
        back_pts = cnt[keep & horiz & (ys < cy)]
        front_pts = cnt[keep & horiz & (ys > cy)]
        left_pts = cnt[keep & ~horiz & (cnt[:, 0] < cx)]
        right_pts = cnt[keep & ~horiz & (cnt[:, 0] > cx)]
        thr = max(2.0, h / 180.0)
        cys, cxs = np.nonzero(carpet[::3, ::3])
        region = np.stack([cxs * 3.0, cys * 3.0], axis=1)
        outward = {"back": np.array([0.0, -1.0]), "left": np.array([-1.0, 0.0]),
                   "right": np.array([1.0, 0.0]), "front": np.array([0.0, 1.0])}
        lines = {}
        for name, pts in (("back", back_pts), ("left", left_pts), ("right", right_pts), ("front", front_pts)):
            if len(pts) < 12:
                lines[name] = None
                continue
            # Side walls: FLOWERS and people cut notches into the carpet, so
            # the side line must be the outer limit of the carpet. (The back
            # line keeps the plain fit: light HIVE parts above the back wall
            # can touch the carpet mask.)
            line = _ransac_outer_line(np, pts, thr, region, outward[name]) if name in ("left", "right") else None
            lines[name] = line if line is not None else _ransac_line(np, pts, thr)
        self.debug["boundary_points"] = {"back": back_pts, "left": left_pts, "right": right_pts, "front": front_pts}
        if lines["back"] is None or lines["left"] is None or lines["right"] is None:
            return None
        # a "front" line must span a reasonable part of the width
        fl = lines.get("front")
        if fl is not None:
            span = float(np.ptp(fl["support"][:, 0]))
            if span < w * 0.25:
                lines["front"] = None
        return lines

    # ── front edge and corners ────────────────────────────────────────────
    def _dark_rail(self, bg, carpet, lines, y_from: float, y_to: float):
        """Strongest dark straight line (a wall's top rail) crossing the field
        between image rows y_from and y_to, between the side lines. Slopes
        around the back line's slope are tried (the front rail is usually not
        exactly parallel to the back line in the image)."""
        cv2, np = self.cv2, self.np
        h, w = bg.shape[:2]
        gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
        left, right = lines["left"], lines["right"]
        bd = lines["back"]["dir"]
        base_slope = bd[1] / bd[0] if abs(bd[0]) > 1e-6 else 0.0
        xs_all = np.arange(0, w, 2.0)
        best = None
        for slope in base_slope + np.arange(-0.08, 0.0801, 0.008):
            for y0 in np.arange(max(1.0, y_from), min(h - 2.0, y_to), 1.0):
                yl = y0 + (xs_all - w / 2) * slope
                xl = _x_on_line(left, y0)
                xr = _x_on_line(right, y0)
                sel = (yl >= 4) & (yl < h - 5) & (xs_all > xl + 4) & (xs_all < xr - 4)
                if sel.sum() < 15:
                    continue
                xs, ys = xs_all[sel].astype(int), yl[sel]
                vals = gray[ys.astype(int), xs]
                above = gray[(ys - 4).astype(int), xs]
                below = gray[(ys + 4).astype(int), xs]
                frac_dark = float(((np.minimum(above, below) - vals) > 18).mean())
                # prefer lines spanning more of the field width
                span = (xs.max() - xs.min()) / max(xr - xl, 1.0)
                score = frac_dark * min(1.0, span)
                if best is None or score > best[0]:
                    best = (score, y0, slope)
        if best is None or best[0] < 0.45:
            return None
        _score, y0, slope = best
        d = np.array([1.0, slope]) / math.hypot(1.0, slope)
        return {"point": np.array([w / 2.0, y0]), "dir": d, "score": best[0]}

    def _homography(self, corners):
        cv2, np = self.cv2, self.np
        tl, tr, br, bl = corners
        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([[0, 0], [FIELD_IN, 0], [FIELD_IN, FIELD_IN], [0, FIELD_IN]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(dst, src)   # field -> image
        return H

    def _scales_at(self, H, fx: float, fy: float):
        np = self.np

        def proj(x, y):
            p = H @ np.array([x, y, 1.0])
            return p[:2] / p[2]
        p0 = proj(fx, fy)
        lateral = np.linalg.norm(proj(fx + 1.0, fy) - p0)
        depth = np.linalg.norm(proj(fx, fy + 1.0) - p0)
        return lateral, depth

    def _wall_px_per_in(self, H, fy: float):
        lateral, depth = self._scales_at(H, FIELD_IN / 2, fy)
        sin_t = min(0.999, depth / max(lateral, 1e-6))
        cos_t = math.sqrt(max(0.0, 1.0 - sin_t * sin_t))
        return lateral * cos_t

    def _corners(self, bg, carpet, lines, overlay_bands):
        np = self.np
        h, w = bg.shape[:2]
        notes = []
        tl = _intersect(np, lines["back"], lines["left"])
        tr = _intersect(np, lines["back"], lines["right"])
        if tl is None or tr is None:
            return None
        ys = np.nonzero(carpet.any(axis=1))[0]
        carpet_bottom = float(ys.max())
        back_y = (tl[1] + tr[1]) / 2.0

        def corners_for(front_line):
            bl = _intersect(np, front_line, lines["left"])
            br = _intersect(np, front_line, lines["right"])
            return (tl, tr, br, bl) if bl is not None and br is not None else None

        # The near floor edge is (almost) never visible from a broadcast
        # camera: the front wall's TOP rail crosses the view and the bottom of
        # the wall is lower (below the frame / under the overlay), with floor
        # visible through the clear wall in between. So the front floor line
        # is placed below the top rail by the wall's projected height.
        rail = self._dark_rail(bg, carpet, lines,
                               back_y + (carpet_bottom - back_y) * 0.55, carpet_bottom + 3)
        back_rail = self._dark_rail(bg, carpet, lines, max(1.0, back_y - 45), back_y - 3)
        wall_h_in = self.WALL_HEIGHT_IN
        if rail is not None:
            method = "front top rail + wall height"

            def front_line_at(d):
                return {"point": rail["point"] + np.array([0.0, d]), "dir": rail["dir"]}
            d = 10.0
            measured_h = None
            for _ in range(40):
                cs = corners_for(front_line_at(d))
                if cs is None:
                    break
                H = self._homography(cs)
                if back_rail is not None:
                    back_px = _y_on_line(lines["back"], w / 2.0) - _y_on_line(back_rail, w / 2.0)
                    per_in_back = self._wall_px_per_in(H, 0.0)
                    if per_in_back > 1e-3 and back_px > 0:
                        measured_h = back_px / per_in_back
                h_use = measured_h if (measured_h is not None and 9.0 <= measured_h <= 15.0) else self.WALL_HEIGHT_IN
                wall_h_in = h_use
                new_d = h_use * self._wall_px_per_in(H, FIELD_IN)
                if abs(new_d - d) < 0.05:
                    d = new_d
                    break
                d = 0.5 * d + 0.5 * new_d
            front = front_line_at(d)
            notes.append("front floor line {:.1f} px below the front top rail (wall {:.1f} in{})".format(
                d, wall_h_in, ", measured at the back wall" if (measured_h is not None and 9.0 <= measured_h <= 15.0) else ", default"))
        elif lines.get("front") is not None and not any(
                abs(_y_on_line(lines["front"], w / 2.0) - y) < 6 for band in overlay_bands for y in band):
            # High / overhead camera: the carpet's front boundary is the floor
            # edge itself (no rail crossing the field).
            front = lines["front"]
            method = "front floor line visible"
        else:
            front = {"point": np.array([w / 2.0, carpet_bottom]), "dir": lines["back"]["dir"]}
            method = "carpet bottom (low confidence)"
            notes.append("no front top rail found; used the bottom of the visible carpet")
        corners = corners_for(front)
        if corners is None:
            return None
        tl, tr, br, bl = corners

        def visible(p):
            x, y = p
            if not (0 <= x < w and 0 <= y < h):
                return False
            return not any(y0 <= y <= y1 for y0, y1 in overlay_bands)
        vis = sum(1 for p in corners if visible(p))
        support = sum(lines[k]["inliers"] for k in ("back", "left", "right") if lines.get(k))
        if method.startswith("carpet bottom"):
            conf = "low"
        elif support > 300:
            conf = "high"
        elif support > 150:
            conf = "medium"
        else:
            conf = "low"
        H = self._homography(corners)
        return {
            "corners": corners, "front_method": method, "visible": vis, "confidence": conf,
            "wall_px_per_in": round(float(self._wall_px_per_in(H, FIELD_IN)), 3),
            "wall_height_in": round(float(wall_h_in), 2), "H": H, "notes": notes,
            "back_rail": back_rail, "front_rail": rail,
        }

    def vertical_px_per_in(self, H, fx: float, fy: float) -> float:
        """Approximate image pixels per inch of HEIGHT above field point (fx, fy)."""
        lateral, depth = self._scales_at(H, fx, fy)
        sin_t = min(0.999, depth / max(lateral, 1e-6))
        return lateral * math.sqrt(max(0.0, 1.0 - sin_t * sin_t))

    def project(self, H, fx: float, fy: float, height_in: float = 0.0):
        np = self.np
        p = H @ np.array([fx, fy, 1.0])
        x, y = p[0] / p[2], p[1] / p[2]
        if height_in:
            y -= height_in * self.vertical_px_per_in(H, fx, fy)
        return float(x), float(y)

    # ── HIVE and FLOWER boxes ─────────────────────────────────────────────
    # HIVE geometry (BIOBUZZ manual 9.6): structure at field center, pivots
    # 43.95 in above the tiles, CELLS 18.8 in apart along the field depth,
    # openings about 20 in wide. Each alliance's HIVE occupies one side of
    # the center line in the camera view (season.HIVE_SIDES).
    HIVE_PRIOR_X_IN = (1.0, 27.0)       # distance from the center line
    HIVE_PRIOR_Y_IN = (-24.0, 24.0)     # along the depth, around center
    HIVE_PRIOR_Z_IN = (20.0, 66.0)      # height above the tiles

    def _hive_prior_box(self, corners, side: str, w: int, h: int):
        np = self.np
        H = corners["H"]
        x0, x1 = self.HIVE_PRIOR_X_IN
        xs = (FIELD_IN / 2 - x1, FIELD_IN / 2 - x0) if side == "left" else (FIELD_IN / 2 + x0, FIELD_IN / 2 + x1)
        pts = []
        for fx in xs:
            for dy in self.HIVE_PRIOR_Y_IN:
                for z in self.HIVE_PRIOR_Z_IN:
                    pts.append(self.project(H, fx, FIELD_IN / 2 + dy, z))
        pts = np.array(pts)
        bx0, by0 = pts.min(axis=0)
        bx1, by1 = pts.max(axis=0)
        return (int(max(0, bx0)), int(max(0, by0)), int(min(w - 1, bx1)), int(min(h - 1, by1)))

    def _hive_boxes(self, frames, corners) -> Optional[Dict[str, list]]:
        """Each HIVE box: the alliance color seen in at least 15% of frames
        (both tip positions) inside a geometric prior box projected from the
        HIVE's known size and position; the prior itself when too little color
        is found."""
        cv2, np = self.cv2, self.np
        h, w = frames[0].shape[:2]
        acc = {"red": np.zeros((h, w), np.uint16), "blue": np.zeros((h, w), np.uint16)}
        for f in frames:
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            for color, ranges in season.HIVE_HSV_RANGES.items():
                if color not in acc:
                    continue
                m = None
                for lo, hi in ranges:
                    part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                    m = part if m is None else cv2.bitwise_or(m, part)
                acc[color] += (m > 0).astype(np.uint16)
        out = {}
        self.debug["hive_priors"] = {}
        for color in ("red", "blue"):
            side = season.HIVE_SIDES.get(color, "left")
            px0, py0, px1, py1 = self._hive_prior_box(corners, side, w, h)
            if px1 - px0 < 6 or py1 - py0 < 6:
                continue
            self.debug["hive_priors"][color] = (px0, py0, px1, py1)
            m = (acc[color] >= max(2, int(0.15 * len(frames)))).astype(np.uint8) * 255
            roi = np.zeros_like(m)
            roi[py0:py1 + 1, px0:px1 + 1] = 255
            m = cv2.bitwise_and(m, roi)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            ys, xs = np.nonzero(m)
            prior_area = (px1 - px0) * (py1 - py0)
            if len(xs) >= 0.01 * prior_area:
                # robust extent of the HIVE color inside the prior
                xa, xb = np.percentile(xs, [1, 99])
                ya, yb = np.percentile(ys, [1, 99])
                pad_x = 0.08 * (xb - xa) + 2
                pad_y = 0.08 * (yb - ya) + 2
                xa, xb = max(px0, xa - pad_x), min(px1, xb + pad_x)
                ya, yb = max(0, ya - pad_y), min(py1, yb + pad_y)
            else:
                xa, ya, xb, yb = px0, py0, px1, py1
            out[color] = [[int(xa), int(ya)], [int(xb), int(ya)], [int(xb), int(yb)], [int(xa), int(yb)]]
        if "red" in out and "blue" in out:
            r, b = out["red"], out["blue"]
            left_box, right_box = (r, b) if season.HIVE_SIDES.get("red") == "left" else (b, r)
            if left_box[1][0] >= right_box[0][0]:
                mid = (left_box[1][0] + right_box[0][0]) // 2
                left_box[1][0] = left_box[2][0] = mid - 1
                right_box[0][0] = right_box[3][0] = mid + 1
        return out or None

    def _flower_boxes(self, bg, corners) -> Optional[List[list]]:
        """FLOWER boxes from their known positions on the field
        (season.FLOWER_POSITIONS_IN): the column from the floor to the top
        ring and backstop. Their yellow rings and POLLEN placed in them are not
        shots, so these boxes are removed from shot detection."""
        np = self.np
        h, w = bg.shape[:2]
        H = corners["H"]
        r = season.FLOWER_RADIUS_IN
        top = season.FLOWER_HEIGHT_IN
        boxes = []
        for fx, fy in season.FLOWER_POSITIONS_IN:
            pts = np.array([self.project(H, fx + dx, fy + dy, z)
                            for dx in (-r, r) for dy in (-r, r) for z in (0.0, top)])
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            pad = 0.15 * (x1 - x0) + 2
            x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
            if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
                continue
            x0, y0 = int(max(0, x0)), int(max(0, y0))
            x1, y1 = int(min(w - 1, x1)), int(min(h - 1, y1))
            boxes.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return boxes or None


def _x_on_line(line, y: float) -> float:
    p, d = line["point"], line["dir"]
    if abs(d[1]) < 1e-9:
        return float(p[0])
    t = (y - p[1]) / d[1]
    return float(p[0] + t * d[0])


def draw_calibration(cv2, np, frame, calib: Dict[str, object]):
    """Preview image: field outline (extended past the frame), HIVE, FLOWER
    and exclusion boxes."""
    img = frame.copy()
    h, w = img.shape[:2]
    pad = max(h, w) // 2
    canvas = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    bl, br, tr, tl = calib["corners_px"]
    pts = np.array([[p[0] + pad, p[1] + pad] for p in (tl, tr, br, bl)], dtype=np.int32)
    cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    for name, p in zip(("TL", "TR", "BR", "BL"), pts):
        cv2.circle(canvas, tuple(int(v) for v in p), 5, (0, 255, 255), -1)
        cv2.putText(canvas, name, (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for color, box in (calib.get("hive_boxes_px") or {}).items():
        b = np.array([[q[0] + pad, q[1] + pad] for q in box], dtype=np.int32)
        cv2.polylines(canvas, [b], True, (40, 40, 255) if color == "red" else (255, 120, 40), 2)
    for box in calib.get("flower_boxes_px") or []:
        b = np.array([[q[0] + pad, q[1] + pad] for q in box], dtype=np.int32)
        cv2.polylines(canvas, [b], True, (0, 200, 0), 1)
    for box in calib.get("exclude_px") or []:
        b = np.array([[q[0] + pad, q[1] + pad] for q in box], dtype=np.int32)
        cv2.polylines(canvas, [b], True, (160, 160, 160), 1)
    cv2.rectangle(canvas, (pad, pad), (pad + w - 1, pad + h - 1), (255, 255, 255), 1)
    # crop to the area that matters
    xs = list(pts[:, 0]) + [pad, pad + w]
    ys = list(pts[:, 1]) + [pad, pad + h]
    x0, x1 = max(0, min(xs) - 10), min(canvas.shape[1], max(xs) + 10)
    y0, y1 = max(0, min(ys) - 10), min(canvas.shape[0], max(ys) + 10)
    return canvas[y0:y1, x0:x1]
