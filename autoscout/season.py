"""
Season-specific configuration.

Everything that depends on the current FTC game (or on the typical camera
setup for that game's broadcasts) lives here, so adapting AutoScout to a new
season starts in this one file.

Current game: BIOBUZZ presented by RTX (2026-2027).
"""

GAME_NAME = "BIOBUZZ"
SEASON_LABEL = "2026-2027"

# Field image used by the dashboard / data visualizer. It must be oriented the
# way the camera sees the field: far wall at the top, audience wall at the
# bottom, red alliance on the left (center-origin field frame, x right, y down).
FIELD_IMAGE = "assets/biobuzz-field.png"

# ── Scoring elements ─────────────────────────────────────────────────────────
# OpenCV HSV ranges (H 0-179, S 0-255, V 0-255) for the scoring element the
# shot detector tracks. BIOBUZZ POLLEN is yellow. Add more (lo, hi) tuples to
# track additional colors.
BALL_HSV_RANGES = [
    ((18, 110, 110), (35, 255, 255)),   # POLLEN (yellow)
]

# ── HIVE / scoring targets ──────────────────────────────────────────────────
# Which side of the image each alliance's HIVE appears on (relative to the
# projected field center). Used only when field_corners.json has no
# hive_boxes_px and the detector has to find the HIVES by color.
HIVE_SIDES = {"red": "left", "blue": "right"}

# Alliance colors of the HIVE structures (OpenCV HSV), used by HIVE tip
# detection (autoscout/hive.py).
HIVE_HSV_RANGES = {
    "red": [((0, 90, 45), (12, 255, 255)), ((170, 90, 45), (179, 255, 255))],
    "blue": [((95, 80, 45), (135, 255, 255))],
}

# FTC match timing (seconds from match start) used to label events:
# 30 s autonomous, 8 s transition, 2:00 teleop.
AUTO_END_S = 30.0
TELEOP_START_S = 38.0
MATCH_END_S = 158.0

# ── Tracker behavior switches ───────────────────────────────────────────────
# DECODE-only: suppress blobs in the two far corners (field structures there).
CORNER_ZONE_SUPPRESSION = False

# Use a robot pixel size that varies with field position instead of one size
# measured at field center. Needed for low, angled cameras where near robots
# look 2-3x larger than far ones. Set False for near-overhead cameras.
PERSPECTIVE_ROBOT_SCALE = True

# Floor-contact correction. A blob's center sits above the robot's footprint
# on angled cameras; before projecting to the field, the point is moved down by
# FOOT_OFFSET_FRAC * (local 18 in robot size in pixels), clamped to the blob's
# bottom edge. 0.0 = old behavior (good for overhead cameras).
FOOT_OFFSET_FRAC = 0.30

# Robots can start hidden (e.g. behind the HIVE). If fewer than 4 robots are
# visible after INIT_PARTIAL_AFTER_FRAMES processed frames, the tracker starts
# with the visible ones (at least INIT_MIN_ROBOTS) and picks up the rest when
# they appear.
INIT_PARTIAL_AFTER_FRAMES = 20
INIT_MIN_ROBOTS = 2

# Robots stand still before the match starts while field staff walk around.
# Only blobs that stayed within INIT_STABLE_IN inches for the last
# INIT_STABLE_FRAMES processed frames can seed the starting lineup.
# 1 disables the check (previous behavior).
INIT_STABLE_FRAMES = 5
INIT_STABLE_IN = 4.0

# A track whose robot has been missing for a while (lost) may not take a blob
# that a currently visible robot is on. Without this, a lost track steals a
# visible robot's blob, the two tracks alternate on that robot and both look
# visible (duplicate robots). False = previous behavior.
PROTECT_VISIBLE_FROM_LOST = True

# Static-blob suppression drops blobs that have not moved for a while unless
# a robot track is nearby. With True, a lost (temporarily invisible) track
# also counts, so a parked robot whose track dropped out for a moment is not
# suppressed forever. False = previous behavior (only visible tracks count).
STATIC_SUPPRESS_SPARES_LOST_TRACKS = True

# When a track is re-acquired far from its last position (it was lost, or
# snapped to a different blob), treat it as a fresh start (zero velocity)
# instead of feeding the jump into its velocity, which otherwise throws the
# next prediction far off and loses the robot again. False = previous behavior.
RESET_VELOCITY_ON_JUMP = True

# Robot blob sizes. The tracker decides how many robots a blob contains by
# comparing its area with an 18 in square at that spot. Real robots on angled,
# blurry video often look much smaller than that (about 0.6x on the Piper
# Meet stream), so two touching robots never reached the 2-robot threshold.
# With True, the tracker measures the typical single-robot blob at
# initialization and scales the thresholds by it (clamped to 0.3-1.2).
AUTO_ROBOT_AREA = True
