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
