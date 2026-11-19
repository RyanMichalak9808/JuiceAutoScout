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

# Rules for lost tracks (robot not seen for more than VISIBLE_COAST frames,
# e.g. hidden behind the HIVE or inside a pile of robots). With True:
# - a lost track may only re-acquire a robot standing on its own (an unsplit
#   blob), never a piece of a merged blob,
# - every track (lost or not) may only take a robot within
#   LOST_REACQ_SPEED_IN_S * (seconds since last seen) + LOST_REACQ_SLACK_IN
#   of where it was last seen (no jumping across the field),
# - its stale position no longer adds "a robot should be here" hints when
#   splitting merged blobs.
# Without these, lost tracks jumped into piles of robots for single frames and
# reappeared as ghosts at old positions (labels swapping between robots).
LOST_TRACK_RULES = True
LOST_REACQ_SPEED_IN_S = 80.0
LOST_REACQ_SLACK_IN = 24.0

# Learn what each robot looks like (color histogram) while it is tracked on its
# own, and use that to decide which track a robot belongs to when it reappears
# or leaves a pile. Off = appearance is only used with --manual-reference-csv.
AUTO_REID = True

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

# Learn a background image of each HIVE for each of its two resting positions,
# so robots driving behind/under the HIVE stay detectable. The HIVE box is only
# removed from robot detection while that HIVE is tipping (or its position is
# unknown). False = HIVE boxes are always removed from robot detection.
HIVE_STATE_BACKGROUND = True

# Two robots in contact: with True they are handled like 3-4 robot piles -
# each track follows its robot's peak inside the merged blob, and on separation
# the left-to-right order along the contact axis decides who is who. The
# original 2-robot logic froze both tracks where the contact started (so a
# robot pushed 30 in during contact was lost on separation) and its swap check
# could never trigger. False = original behavior.
FOLLOW_TWO_ROBOT_MERGES = True

# Identity correction by appearance: each robot's look is learned from its
# first isolated samples (AUTO_REID early bank). While robots are on their own,
# the tracker keeps comparing each tracked robot with every robot's learned
# look; if the labels are consistently better explained by a different
# assignment (two labels swapped, three rotated, or a robot that belongs to a
# currently hidden label), the labels are moved back. Fixes swaps after
# robots were hidden or merged together. Needs AUTO_REID.
APPEARANCE_ID_CORRECTION = True

# Robot color fingerprints (AUTO_REID) use hue and saturation only, so
# brightness changes (shadows, venue lighting) do not change them. With
# REID_CORE_FRAC set, only the inner part of the robot's outline is sampled
# (the given fraction of its area, deepest pixels first), so grey floor tiles
# bleeding into the outline edges do not wash out the robot's colors.
# None = sample the whole outline (previous behavior).
# Tested on the Piper Meet example (640x360): whole outline identified robots
# 16/18 times, inner 80% 15/18, 65% 14/18, 50% 13/18 - at this resolution a
# robot is only 30-40 px wide and its edges (bumpers, plates) carry the most
# distinctive color, while the foreground mask already excludes most floor.
# Off by default; worth re-testing on higher-resolution footage.
REID_CORE_FRAC = None

# Join pieces of one robot split by a thin horizontal bar in front of it (the
# top rail of the front field wall cuts robots parked against it in two).
# Vertical closing height in pixels at 640 px wide video; 0 = off.
FG_VERTICAL_CLOSE_PX = 7
# Along the front wall the rail is thicker in the image (closer to the
# camera), so a stronger vertical join is used in a band around the front
# wall line only (height of the join and half-height of the band, px at 640).
FRONT_RAIL_CLOSE_PX = 16
FRONT_RAIL_BAND_PX = 18

# Ignore foreground blobs bigger than this many robots (a referee or
# spectator standing in front of the camera) instead of splitting them into
# "robots". 0 = off.
MAX_BLOB_ROBOTS = 6.0

# AUTO_ROBOT_AREA: blobs whose size is implausible for one robot (in units of
# an 18 in square) are ignored when measuring the single-robot size, and the
# measurement is refined from the first AUTO_ROBOT_AREA_SAMPLES robots seen
# clearly on their own.
AUTO_ROBOT_AREA_RANGE = (0.25, 1.6)
AUTO_ROBOT_AREA_SAMPLES = 30

# Inside a pile of robots (merge group) a label only counts as seen when a
# real robot peak in the merged blob is matched to it. A label with no peak
# for this many seconds leaves the pile and becomes lost (so it can pick its
# robot up again elsewhere) instead of staying frozen and "visible" inside a
# pile its robot already left. 0 = off (original: all labels in a pile stay
# visible, frozen at their last peak).
# Off by default: on the Shanghai example the released label rejoined the
# same pile a moment later (release/rejoin cycles), and overall label jumps
# and duplicates got slightly worse on all three example videos.
MERGE_UNSEEN_RELEASE_S = 0.0

# Detect the field automatically (autoscout/field_detect.py) when no
# --corners file is given. The dashboard shows the result for approval
# before a match is processed. False = old edge-based detector.
AUTO_FIELD_CALIBRATION = True

# People off the field (referees, drive teams, a referee standing in front of
# the camera): foreground blobs with at least PERSON_OUTSIDE_FRAC of their
# area outside the field walls (left, right or in front; above the back wall
# is allowed because robots stick up there) and at least
# PERSON_MIN_ROBOT_AREAS robot-areas outside are removed from robot AND ball
# detection - including the part of them that overlaps the field.
FILTER_OFF_FIELD_PEOPLE = True
PERSON_OUTSIDE_FRAC = 0.5
PERSON_MIN_ROBOT_AREAS = 0.8

# FLOWER positions (field coordinates in inches, origin at the far-left
# corner, x to the right, y toward the camera - the same frame as the field
# image) and size, used to place FLOWER boxes during automatic calibration.
FLOWER_POSITIONS_IN = [(48.1, 2.5), (141.5, 48.1), (2.5, 95.8), (95.8, 141.5)]
FLOWER_RADIUS_IN = 3.5
FLOWER_HEIGHT_IN = 24.0

# Frames where more than this fraction of the field differs from the
# background (camera cut, flash, broadcast transition) are skipped.
GLITCH_FIELD_CHANGED_FRAC = 0.55
# Foreground blobs that reach the bottom of the picture and whose part inside
# the field area is taller than this many robot sizes (measured where they
# touch the floor) are people standing in front of the camera. 0 = off.
PERSON_MAX_ROBOT_HEIGHTS = 2.6

# Processing window: robot, ball and people detection run only on a crop
# around the field and HIVES plus ROI_MARGIN_FRAC of the frame width on each
# side (people at the walls must stay partly visible to be recognized). The
# crowd and overlays outside are never processed; debug videos still show
# the whole stream. False = process the whole frame.
PROCESS_ROI = True
ROI_MARGIN_FRAC = 0.06
