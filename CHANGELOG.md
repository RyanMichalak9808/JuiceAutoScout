# Changelog: BIOBUZZ (2026-27) adaptation

This file is a running summary of the changes made to AutoScout to adapt it from DECODE (2025-26) to BIOBUZZ (2026-27). For each change it records what it is, why it was needed, and the reasoning behind it. It also records how big the change is compared with the original repository (commit `4d1be90`) and how much more processing a match now needs.

## At a glance

- **New season:** POLLEN (yellow) shots into the red / blue HIVE, HIVE tip counting, and the BIOBUZZ field image.
- **New camera:** a low, audience-side event stream (640x360, blurry) instead of the high overhead DECODE World Championship broadcast.
- **Robot labels:** the main problem on the test videos was robot labels swapping or disappearing after collisions and behind the HIVE. Most of the tracker work went there.
- **Pause / resume:** processing can be paused, the dashboard shut down, and resumed later; finished matches and approved calibrations are kept, and interrupted matches are redone from the start.
- **Scouting analytics:** per-alliance tip timing (AUTO tips, cycle pace, endgame behavior), per-robot movement red flags, a shot heatmap and a loose game-piece density map, written for every match.
- **Shots and setup:** the shot detector was rewritten for steep, vertical shots and now follows balls on every frame. Field corners, HIVE boxes and FLOWER boxes are detected automatically for every match, and the dashboard asks for approval of that calibration before tracking starts. Referees and people off the field are filtered out of robot and ball detection.
- **Original behavior preserved:** every new behavior is behind a switch in `autoscout/season.py`. With the switches set back to their DECODE values, the tracker reproduces the original DECODE output exactly (0 differences over 6,128 rows of `examples/joosmatch.mp4`, rechecked after every round).

## Size of the change compared with the original repo

Measured with `git diff 4d1be90 HEAD`. Example videos and images are excluded.

| | Original | Now | Change |
|---|---|---|---|
| Code lines (`.py`, `.js`, `.html`) | 14,489 | 17,693 | **+3,729 / -525 lines, about 29% of the original code size touched; net +22%** |
| Original lines removed or replaced | | | 525 (3.6% of the original code) |
| Python lines | 8,025 | 10,828 | +35% |
| `autoscout/tracker.py` | 1,490 | 2,347 | +58% |
| `autoscout/shot.py` | 470 | 536 | +14% (most of the logic rewritten) |
| `auto_scout.py` | 809 | 1,061 | +31% |
| `autoscout/dashboard_server.py` | 950 | 1,132 | +19% |
| New files | | | `autoscout/season.py`, `autoscout/analytics.py`, `autoscout/scouting_stats.py`, `autoscout/stream_split.py`, `autoscout/field_detect.py`, `autoscout/hive.py`, `autoscout/match_start.py`, `dashboard_static/calib_editor.js`, `tools/find_match_start.py`, `assets/biobuzz-field.png`, `CHANGELOG.md`, three example videos and their calibrations |

Most of the work is added code behind switches; apart from the shot detector, little original code was rewritten.

## Processing cost per match

The original and new versions were timed on the same machine (4 cores), one after the other, without `--debug-video`.

**Update (section 10):** with the processing window, reused buffers and small masks, q9 takes 54.8 s, about 7% faster than the original code (59.0 s). The measurements below are from before that change.

Measurement after the shot detector rewrite, `piper_q9.mp4` (BIOBUZZ, 3 min, 640x360):

| Run | Wall time | CPU time (user + sys) |
|---|---|---|
| Original | 59.0 s | 126.6 s |
| New, with a calibration file | 85.4 s (**+45%**) | 171.4 s (**+35%**) |
| New, automatic calibration (no file) | 88.0 s (+49%) | 175.9 s (+39%) |

Before the shot detector rewrite the overhead was +11 to +14% wall time and +6 to +8% CPU on the same video; the DECODE example was +8% wall, +0% CPU.

Where the extra time goes:

- **Following balls on every frame** (the largest share): a ball-only pass runs on the 2 of every 3 frames the robot tracker skips, so fast vertical shots can be followed.
- The off-field people filter and the camera-cut check, every frame.
- HIVE tip detection, about 1 ms per frame.
- Per-position robot size and floor-contact projection (small homography calls).
- Robot appearance histograms for re-identification, plus the identity check.
- Learning the two HIVE backgrounds and the automatic field calibration, once per match (about 3 s).

`--debug-video` adds roughly the same overhead to both versions. `tools/find_match_start.py` is a separate helper and only runs when you call it (about 10-20 s per video).

## What changed, why, and the reasoning

### 1. Season adaptation (first commit)

**Game settings in one place: `autoscout/season.py`**
- **What:** colors, HIVE sides, field image, match timing and every behavior switch live in this one file.
- **Why:** the paper says the shot layer is per-game and the tracker needs retuning per season.
- **Reasoning:** keeping all of that in one file makes the next season a one-file job and keeps the DECODE behavior reproducible.

**Calibration (`tools/calibrate.py`, `field_corners.json`)**
- **What:** you can now draw HIVE boxes (key `H`) and exclusion boxes (key `X`). An existing calibration file is reloaded so it can be adjusted, and saving corners from the dashboard keeps these extra fields.
- **Why:** the HIVE is about 44 in above the floor, so the floor-plane homography cannot locate it. The on-screen clock and overlays also create false robots.
- **Reasoning:** two clicks per box is reliable for every camera setup. Detecting these by color was not.

**Perspective-aware robot size (tracker)**
- **What:** the 18 in robot size is computed at each robot's own position instead of once at field center.
- **Why:** on a low camera a near robot looks 2-3x larger than a far one. One size made near robots look like 2-3 merged robots and made far robots too small to detect.
- **Reasoning:** the size comes from the same homography the tracker already uses, so no new tuning was needed.

**Floor-contact point (tracker)**
- **What:** each blob center is moved down by `FOOT_OFFSET_FRAC` (0.30) of the local robot size before it is projected to the field.
- **Why:** from an angle, the center of a robot's silhouette sits above its footprint, which pushed every position toward the far wall.

**Robots hidden at the start (tracker)**
- **What:** if fewer than 4 robots are visible after 20 frames, tracking starts with the visible ones and the rest are picked up when they appear.
- **Why:** BIOBUZZ robots can start behind the HIVE.

**POLLEN mask (tracker)**
- **What:** a yellow-only color mask, plus a separate "moving yellow" mask for shot detection. The shot mask covers the field and the space above the far wall where the HIVES are.
- **Why:** shots fly above the field area that the original mask was limited to. Yellow that never moves (FLOWER rings, POLLEN piles, the HIVE frame panel) must be ignored.
- **Reasoning:** red and blue NECTAR were deliberately not added, because those colors match robots' alliance markers.

**Shot detection (`autoscout/shot.py`)**
- **What:**
  - targets are the calibrated HIVE boxes;
  - the color fallback uses the HIVE sides from `season.py`;
  - a new "made" rule counts a ball that comes to rest inside the HIVE;
  - thresholds scale with frame width.
- **Why:** the HIVE cells are clear, so a made ball stays visible instead of disappearing. The DECODE goal geometry did not apply.

**Field image**
- **What:** `assets/biobuzz-field.png` is the provided diagram rotated 90 degrees counter-clockwise to match the camera view. It was checked against the camera's loading zones, gardens, HIVES and FLOWERS.

### 2. Fixes from the first test video (second commit)

**Starting lineup**
- **What:** only blobs that stay still for 5 frames can be a starting robot, and each must be its own blob.
- **Why:** a staff member walking on the field before the match became a robot track.

**Robots at the front wall**
- **What:** floor-contact positions slightly outside the field are clamped onto the edge.
- **Why:** robots at the front wall were being thrown away as off-field.

**Lost labels taking visible robots**
- **What:** a lost track may not take a blob that a visible robot is on.
- **Why:** a lost track would take a visible robot's blob, and the two labels then alternated on one robot. That caused duplicate labels in about half of all frames.

**Parked robots**
- **What:** static-blob suppression now spares parked robots whose track briefly dropped out.
- **Why:** those robots were otherwise suppressed for the rest of the match.

**Re-acquisition jumps**
- **What:** a big re-acquisition jump resets the robot's velocity.
- **Why:** the jump was being read as speed, which threw the next prediction far off and lost the robot again.

**Smaller HIVE approach zone**
- **What:** the "approaching the HIVE" zone around hand-drawn boxes is smaller.
- **Why:** it reached mid-field and turned stray balls into "missed" shots.

### 3. HIVE tips, event logs, robot size (third commit)

**HIVE tip detection (`autoscout/hive.py`)**
- **What:** tracks where the HIVE's red or blue sits inside its box. That height jumps when the HIVE tips, so a jump that holds for 1 s counts as a tip.
- **Why:** tips were the stated top priority.
- **Reasoning:** the HIVE is a two-position seesaw, so this signal is clean and doesn't depend on seeing the shots. It matched all 13 tips on the two example videos and is based on the HIVE physically moving, not on the scoreboard.

**Event logs**
- **What:** `hive_tips.csv` and `shot_events.csv`, with exact video time, match time, phase and an approximate broadcast clock, plus a summary printed at the end of each run.
- **Why:** so tips and shots can be compared by hand.

**Match timing**
- **What:** 30 s auto + 8 s transition + 2:00 teleop.
- **Why:** the 8 s transition was originally missing.

**Single-robot size learned at start**
- **What:** the tracker measures how big a single robot looks at the start and scales its thresholds by that.
- **Why:** on this stream a single robot measured about 0.6x the 18 in square, so two touching robots never looked big enough to split.

**Shot detector tweaks**
- **What:**
  - the maximum jump for a launched ball is tighter;
  - balls inside a HIVE while it tips cannot count as shots.
- **Why:** unrelated POLLEN was being chained into fake shots.

**Debug overlay**
- **What:**
  - speed arrows show real speed in in/s;
  - robot shapes with no label get a `no track` tag;
  - each HIVE shows its tip count.
- **Why:** arrows used to disappear and showed no magnitude.

### 4. Keeping robot labels (fourth commit)

**Reach limit (`LOST_TRACK_RULES`)**
- **What:**
  - a label can only move to a robot it could have driven to since it was last seen;
  - lost labels only re-acquire robots standing on their own;
  - lost labels no longer add stale "a robot should be here" hints when splitting piles.
- **Why:** hidden labels jumped into piles of robots for a frame and came back as ghosts. This was most of the flicker in the first video.

**Following touching robots (`FOLLOW_TWO_ROBOT_MERGES`)**
- **What:** two touching robots are followed through the contact, and their order along the contact axis decides who is who when they separate.
- **Why:** the original froze both labels where the contact started, so a robot pushed 30 in lost its label. The original swap check could also never trigger.

**HIVE background per position (`HIVE_STATE_BACKGROUND`)**
- **What:** the tracker learns what each HIVE box looks like in both HIVE positions, and only blocks the box while the HIVE is tipping.
- **Why:** blocking the whole HIVE box hid a strip about 30 in deep at the back-center of the field, so robots disappeared "behind the HIVE" even when they were visible.

**Appearance re-ID and identity correction (`AUTO_REID`, `APPEARANCE_ID_CORRECTION`)**
- **What:**
  - each robot's colors (hue and saturation) are learned while it is on its own;
  - if the labels are consistently better explained by a swap or rotation, they are moved back.
- **Why:** once a label was on the wrong robot, nothing could fix it. In q9 three labels rotated between robots hidden behind the red HIVE (34-46 s); this correction now fixes it.
- **Reasoning:** on frames with known identities, nearest-color matching picked the right robot 16 out of 18 times. Corrections need about 6 consistent frames, which avoids flip-flopping.

**Hidden robots in the debug video**
- **What:** a hidden robot keeps a hollow `R#?` marker at its last known spot.

**Start-time helper (`tools/find_match_start.py`)**
- **What:** finds start candidates from the video itself (first robot to drive, first POLLEN moving in a HIVE) and saves frames around each to confirm.
- **Why:** the clock and scoreboard are not reliable, so nothing in the pipeline reads them.
- **Reasoning:** neither cue is reliable alone on these streams, so the tool suggests candidates instead of setting the start automatically.

### 5. Color-based robot identification options (fifth commit)

**Suggestion:** track each robot by its colors as a backup when tracking is lost or jumps. Use HSV with brightness (V) down-weighted, and sample the inner 60-70% of the robot so floor at the edges doesn't wash the colors out.

**What already existed:** `AUTO_REID` and `APPEARANCE_ID_CORRECTION` (section 4) already do this kind of color backup. Each robot's colors are learned while it is on its own and used when it reappears or when labels look swapped. The fingerprint uses hue and saturation only; brightness is ignored entirely, so shadows do not change it.

**Added:** a `REID_CORE_FRAC` switch that samples only the inner fraction of each robot's outline.

**Tested** on frames of the q9 example where the identity of every robot is known (18 robot samples, each matched to the nearest sample of a robot from other frames):

| Fingerprint | Correct robot | Margin (other robot - own robot) |
|---|---|---|
| Hue + saturation, whole outline (current) | **16 / 18** | 0.107 |
| Inner 80% of the outline | 15 / 18 | 0.095 |
| Inner 65% | 14 / 18 | 0.088 |
| Inner 50% | 13 / 18 | 0.078 |
| + brightness at weight 0.15 | 16 / 18 | 0.100 |
| + brightness at weight 0.30 | 15 / 18 | 0.094 |

**Reasoning:**
- At 640x360 a robot is only 30-40 px wide, so its edges (bumpers, number plates) carry the most distinctive colors.
- The outline comes from the foreground mask, which already excludes most of the floor.
- Adding brightness back makes matching slightly worse.

So the current fingerprint stays as it is. Core sampling is kept as a switch, off by default, to re-test on higher-resolution footage, where floor bleed is a bigger share of the outline.

### 6. Third test video: China Shanghai #2 Qualifier, Playoff Match 5 (sixth commit)

**About the video:**
- **Camera and overlay:** a different venue and camera: lower and wider, with the overlay along the top.
- **Video:** 25 fps.
- **Crowding:** many more tips (red 13, blue 8), people in yellow vests behind the field, and a referee walking in front of the camera.
- **Files:** added as `examples/shanghai_q2_playoff5.mp4` with its calibration (overlay band excluded).
- **Start offset:** 1.7 s, taken from robot motion. The robots start driving about 1 s before the on-screen clock appears.

**Results:**
- **HIVE tips:** 13 red, 8 blue, exactly the final broadcast counters. Three more after the match are labeled `after_end` (field reset).
- **Robot tracking:** worse than on the Piper videos at first: duplicate labels in 5.7% of rows, 275 impossible label jumps.

**Fixes:**
- **Front-rail splits:** robots parked against the front wall were cut in two by the wall's top rail. Fix: a small vertical join (`FG_VERTICAL_CLOSE_PX`) plus a stronger one only in a band along the front wall line (`FRONT_RAIL_CLOSE_PX`, `FRONT_RAIL_BAND_PX`).
- **Robot size measurement:** the single-robot size measured from those pieces and from people at the field edge was 0.30x, the lowest allowed. That made real robots look like two. Fix: implausible shapes are ignored (`AUTO_ROBOT_AREA_RANGE`), and the size is refined from the first robots tracked clearly on their own (`AUTO_ROBOT_AREA_SAMPLES`). It now measures 0.77x on this camera.
- **Starting lineup:** it put two labels 2 in apart on the two pieces of one robot. Fix: starting robots must be at least one robot width (18 in) apart.
- **Huge shapes:** shapes bigger than 6 robots (a person in front of the camera) are no longer split into "robots" (`MAX_BLOB_ROBOTS`).
- **Tried and left off:** labels inside a pile that get no robot peak for 2 s are released (`MERGE_UNSEEN_RELEASE_S`). It is off by default: the released label rejoined the same pile a moment later, and overall jumps and duplicates got slightly worse on all three videos.

**After the fixes:** Shanghai duplicate labels 5.7% -> 1.4%, label jumps 275 -> 174. On the Piper videos the numbers stayed about the same (q9 53 jumps / 0.7% duplicates, q11 56 / 0.9%). All tips are still exact on all three videos.

**Still wrong in Shanghai:**
- **Stuck label:** from about 14 to 38 s one label (R3) stays stuck next to a large robot near the camera at the front-right, while the fourth robot is hidden behind the blue HIVE.
- **Referee:** when the referee walks onto the edge of the field area (around 50 s), labels can latch onto them for a moment.

### 7. Shot detector rewrite, automatic field calibration, approval step (seventh commit)

**Automatic field calibration (`autoscout/field_detect.py`)**
- **What:** corners are found at FLOOR level (where the carpet meets the bottom of the walls), together with the HIVE, FLOWER and overlay boxes. It runs per match, so fields with different camera angles each get their own calibration.
- **Why:** hand-placed corners were wrong. On the Piper videos the bottom corners had been set on the top of the front wall rail instead of its floor line, and the top corners were a little wide.
- **How:**
  - the field carpet is segmented from a median background (pixels that look like carpet in at least 20% of frames count, so parked robots don't hide the floor);
  - the back, left and right floor lines are fitted to the carpet boundary (side lines must be the outer limit of the carpet, so FLOWERS and people at the walls don't pull them inward);
  - lines are extended to corners that are off-screen or under the overlay;
  - the near floor edge is usually hidden behind the front wall's top rail, so the rail is found and the floor line placed below it by the projected wall height. The wall height is measured at the back wall, where floor edge and rail are both visible (12.2-12.3 in measured on q9).
- **HIVE boxes:** a geometric prior from the HIVE's size and position (manual 9.6), refined by the HIVE colors over the match and clipped to stay out of overlay bands.
- **FLOWER boxes:** projected from their known field positions (`season.FLOWER_POSITIONS_IN`).
- **Tested on 6 videos (3 BIOBUZZ, 3 DECODE):**
  - works with 2/4, 3/4 or 4/4 corners visible;
  - known-position checks (FLOWER bases and tops, the tile grid) line up on q9, q11 and Shanghai;
  - on the overhead Edison camera the front edge is under the overlay, so it reports **low** confidence and asks for a check.
- **Fallback:** if detection fails, tracking falls back to the old per-frame outline detector.

**Approval before processing (dashboard)**
- **What:** a match started without a calibration file first runs detection only (`auto_scout.py --calibrate-only`), then waits as **Needs Approval**.
- **The panel** shows a frame from the match with the field outline (gray margin for off-screen corners), HIVE, FLOWER and ignore boxes, all draggable, plus **Approve & Start Tracking**.
- **On approval:** the edited calibration is saved as that match's `field_corners.json` and tracking starts. This is also the per-match manual fallback.
- **Turning it off:** setting `require_calibration_approval` to false starts tracking directly on the automatic calibration.

**Browser calibration tab**
- **Auto-Detect Field** button.
- **Boxes:** +Red HIVE / +Blue HIVE / +FLOWER / +Ignore Area, plus delete.
- **Corners** can be dragged outside the picture.
- **Saving** writes all boxes; `tools/calibrate.py` gained `F` (FLOWER box), `A` (auto-detect) and a gray border for off-screen corners.

**Shot detector rewrite (`autoscout/shot.py`)**
- **Why:** most shots were drawn as gray (not-a-shot) traces and few counted as made. The ball was only looked at on tracker frames (every 3rd frame), so a fast vertical shot jumped too far between looks to stay one track.
- **Every frame:** balls are now followed on every source frame (a light ball-only pass runs on frames the robot tracker skips).
- **Prediction:** includes gravity, and the association window is taller than wide.
- **Shot rule:** a ball rising at least 14 px, mostly in upward steps, sideways at most 1.5x its rise. Shots this season are steep.
- **Made:** the ball reaches the CELL zone (upper 80% of the HIVE box) and is not seen falling back out below the box; it can also settle in the CELL.
- **Missed:** the ball rises to CELL height but doesn't go in, or falls back out.
- **Shooter:** the robot nearest the launch point within 1.6 robot sizes, else `unknown`.
- **Burst fire:** quick bursts are no longer merged by a 0.35 s per-robot time filter.
- **Debug colors:** yellow = shot in flight, green = made, red = missed, gray = ball not judged a shot.

**Filtering things that are not on the field**
- **People:** foreground shapes mostly outside the side or front walls (referees, drive teams) are removed from robot and ball detection, including the part of them above where they stand. Robots touching the crowd beside a wall are kept.
- **People in front of the camera:** shapes that reach the bottom of the picture and are much taller than a robot are removed.
- **Yellow jackets:** POLLEN must be ball-sized and compact; yellow on people is removed with them.
- **FLOWER boxes:** removed from shot detection, so their yellow top rings and POLLEN placed in them are ignored. HIVE boxes always stay searchable.
- **Camera cuts and flashes:** frames where every quarter of the field changes at once are skipped (only after the match ended on the examples).

**Results with automatic calibration**
- **HIVE tips:** still exact on all three BIOBUZZ videos (34 / 34).
- **Hand check:** on q9's first volley (3 POLLEN into the red HIVE at 10.7-10.9 s) all three shots were found from the right robot.
- **Limit:** two or three balls flying up side by side at 640x360 are sometimes counted as one or two.

**How it was checked:**
- **Reviews:** three review passes over all changed files. They found and fixed an unused variable, a duplicate-shot check that could drop a real shot when two robots launched in the same frame, and a missing fallback when automatic calibration fails.
- **Dashboard:** tested in a real browser: Auto-Detect, drawing boxes, saving, the approval panel with a dragged corner and an added box, and the tracking run that followed.
- **Old behavior:** the DECODE regression (switches off) is still 0 differences over 6,128 rows.

**Not yet handled:** a referee's legs that stand inside the field area and don't reach the bottom of the picture can still be detected as foreground. Tracks won't jump to them across the field, and shapes bigger than 6 robots are ignored.

### 8. Match analytics, shot heatmap and resource density (eighth commit)

**Why:** with roughly 70% accurate detection, single events are noisy, but per-match aggregates compared across a team's 5-6 qualification matches still separate a robot that carries its alliance from one whose numbers come from lucky partner pairings, and they show unreliable robots quickly.

**New `autoscout/analytics.py`, written at the end of every run:**
- **`alliance_summary.csv`** (from the timestamped tip log):
  - AUTO tips (first 30 s): teams with a real autonomous HIVE routine;
  - median and mean seconds between tips: scoring pace; still meaningful when a tip in the middle is missed;
  - tips in the final 60 s and final 15 s: a drop after the FLOWERS unlock suggests a FLOWER / Garden pivot, and tips in the last 15 s mark full-match HIVE cyclers.
- **`robot_summary.csv`:** distance, % of time moving, longest stall, field coverage, time on each half, shots made / missed, and red flags: a stall of 30 s or more, never leaving one half, barely moving. Positions jumping faster than 150 in/s and time while hidden are ignored.
- **`shot_heatmap.png`:** shooter positions on the BIOBUZZ field image, made green / missed red, with a density glow.
- **`resource_density.csv` / `.png`:** every 2 s, loose POLLEN (yellow) and red / blue NECTAR are found on the floor (small round ball-sized blobs; robots, people and FLOWER boxes are left out, since red / blue are also bumper colors). They're drawn on the field map with the share of POLLEN on each half early vs late in the match, showing which side runs dry first.
- **Run summary:** the end-of-run printout lists each alliance's tip pattern and any robot red flags.

**Field image:** the standalone data visualizer now loads the BIOBUZZ field image from either location (opened from `tools/` or served by the dashboard). The dashboard already used it.

**Checked:**
- q9 and Shanghai produce all files;
- Shanghai: red median 11 s between tips, blue 21 s;
- q9: no red flags with a 30 s stall limit (robots parked to shoot for about 20 s);
- the DECODE regression is still 0 differences.

**Limits:**
- R0-R3 are tracker labels, not teams.
- Resource detection is approximate: red / blue items near walls can still be miscounted as NECTAR.
- Cost: about 20 ms every 2 s for resource sampling plus about 1 s at the end.

### 9. Pause and resume processing across shutdowns (ninth commit)

**What:** match processing can be paused, the dashboard and computer shut down, and processing resumed later without losing finished work or approvals. Resuming happens per match: a match interrupted halfway is processed again from the start (about 90 s), because saving the full tracker state mid-match would be large and fragile.

**How:**
- **Pause:** the overview's automatic-processing button now pauses as well as starts. Pausing stops new matches from starting. If matches are running, it asks whether to stop them now (they're redone from the start after resuming) or let them finish.
- **Markers in each match folder:**
  - `job_payload.json`: the request that started the match;
  - `running.json`: present while a run is in progress;
  - `done.json`: written by `auto_scout.py` only when a match finishes.
- **Processed status:** a match counts as processed only with `done.json`. Before, a half-written `robot_positions.csv` could make an interrupted match look processed. Older folders without these markers are still accepted.
- **Restart:** matches that were waiting for calibration approval come back as **Needs Approval**.
- **Resume:** a match with an approved `field_corners.json` but no `done.json` is restarted by the auto-queue and reuses that calibration without asking again. A downloaded video is kept until its match finishes, so it isn't downloaded again.

**Checked end to end against the server:**
- a match waiting for approval was restored after a restart;
- after approval, a run stopped halfway was not counted as processed;
- after another restart, starting it again reused the approved calibration (no approval prompt) and ran to completion, and only then counted as processed.

**Note:** "Reprocess Match" also reuses the approved calibration. To recalibrate a match, delete its `field_corners.json` first.

### 10. Faster processing: field window, reused buffers, small masks (tenth commit)

**What:**
- **Processing window:** robot, ball and people detection now run only on a crop around the field and HIVES plus a margin of 6% of the frame width (`season.PROCESS_ROI`, `ROI_MARGIN_FRAC`). The margin keeps people at the walls partly visible so the people filter can still recognize them. The crowd and overlays outside the window are never processed. All masks stay full-frame size (zero outside the window), so the debug videos still show the entire stream.
- **Reused buffers:** the per-frame work images (background difference, gray, threshold, HSV, ball masks, foreground) are allocated once and reused, with OpenCV writing into them in place.
- **Background cache:** the background image for each combination of HIVE positions is built once and cached, instead of copied every frame.
- **Small masks:** each candidate shape's mask (used to split touching robots and measure shape size) is now a small window around that shape, clipped at the picture edge, instead of a full-frame image several times per frame.

**HIVE tip checks:** stay at 10 per second (every robot-tracking frame). A 5-per-second version was tried and dropped: with fewer samples, tip start times drifted by up to a few seconds and a tip right at the buzzer could land after the match end.

**Results (same machine, one run after another):**

| Video | Before | After | Faster by |
|---|---|---|---|
| q9 | 84.6 s | 54.8 s | 35% |
| Shanghai | 88.3 s | 55.8 s | 37% |
| q11 | ~84 s | 52.7 s | ~37% |

On these cameras the field fills most of the picture (the window is 86-90% of the frame), so most of the gain comes from the reused buffers and small masks. A camera that shows more crowd gains more from the window.

**Checked:**
- **Identical output:** tips, shots and robot positions are identical to the previous version on all three BIOBUZZ videos, and DECODE is still identical to the original with the window on or off.
- **A bug found and fixed while checking:** the first version of the small masks padded robots touching the bottom of the picture with a blank border, which changed their centers slightly. The masks are now clipped at the picture edge, which brought Shanghai back to exactly the previous output.

**Processing cost vs the original repo now:** q9 59.0 s originally vs 54.8 s now, so the full BIOBUZZ version with every-frame shot tracking is now about 7% *faster* than the original DECODE code.

### 11. Analytics tab: team stats from FTC Events (eleventh commit)

**What:** a new **Analytics** tab in the dashboard (Overview, Calibration, Settings, Analytics). Every time an event is pulled from FTC Events, the team's scouting calculator (`scoutingcalculatorfixed - 5-3-26.py`) runs on the played qualification matches, and the results are shown there.

**How:**
- **`autoscout/scouting_stats.py`** is a port of the calculator's math:
  - OPR: least squares on alliance scores;
  - EPA: ridge regression on score margins, lambda 10;
  - RSI: OPR as a z-score;
  - oprSOS and epaSOS: mean OPR / EPA of the opponents faced;
  - Record and Win%.
- **Removed:** DPR and CCWM, as requested.
- **Input:** instead of pasted text, it reads the match results (teams and scores) that event loading already pulls from FTC Events. Matches without scores yet are skipped.
- **When it runs:** the server runs it as part of `POST /api/event/discover` and returns `analytics` with the event, so it is recomputed every time the event is loaded.
- **The tab:** a table of all teams with a rank-by selector (EPA, OPR, RSI, oprSOS, epaSOS, Win%; epaSOS ranks lowest first, since lower means a harder schedule), plus an OPR vs Win% scatter chart with team labels.

**Checked:**
- **Same results as the original script:** the same 30 random matches went through both, and EPA, OPR, RSI, oprSOS, epaSOS, Record and Win% matched for all 14 teams (to the script's 2-decimal output).
- **Browser:** the tab was tested with that event loaded: ranking, re-sorting and the chart worked with no page errors.
- **Not tested live:** this build environment can't reach ftc-events.firstinspires.org, so a live event pull wasn't run here. It uses the same results the dashboard already scrapes.

**Note:** the stats are only as good as the scraped scores. A match whose scores can't be read from the FTC Events page is left out.

### 12. Split a full event stream into matches (twelfth commit)

**Why:** many events only post one long stream instead of a video per match, so the dashboard had nothing to process even though the FTC Events results and team stats loaded fine.

**What:** an optional **Full stream YouTube link** field on the Load Event screen, right below the FTC Events URL. If it's filled in, the stream is split as soon as the event loads. The same split can also be started later from the **Full Stream** box on the Overview page. Paste the event's stream (a YouTube link or a video file on this computer) and press **Split Stream**. Every qualification match is found, labeled and cut into its own clip, and those matches can then be processed and queued like any other.

**How it works** (`autoscout/stream_split.py`; kept light, since it mostly works from the audio):
1. **Audio only first.** For a YouTube link, only the audio is downloaded (smallest format, about 20 MB per hour). It is decoded at 8 kHz mono in parallel chunks across the CPU cores.
2. **Field buzzers.** The end-of-AUTO and end-of-match buzzers are a narrow 3240 Hz tone, and they always come exactly 128 s apart (8 s transition + 2:00 TELEOP).
   - Commentary: a narrow-band tone test ignores speech.
   - Stray beeps: pairing at 128 ± 1.5 s rejects them.
   - A match where one buzzer is drowned out is still kept if the other is strong, and is marked low confidence.
   - AUTO starts 29.6 s before the end-of-AUTO buzzer. This was measured from the start sound and from robot motion on the test stream.
3. **Labels from FTC Events.** The start times on FTC Events are only to the minute and are often off, so only the gaps between matches are used.
   - The best single offset is found first, then the two lists are lined up in order.
   - Skips are allowed both ways: replays and false hits in the stream, or matches missing from it.
   - The event page scrape now also records each match's start time (`start_time`).
4. **Clips without re-encoding.** Each match is cut with stream copy from a local file, or downloaded as a yt-dlp section from YouTube (video only up to 720p), with 16 s before AUTO and 10 s after the match (see section 14).
   - Stream-copy cuts start at the previous keyframe, so the buzzer is found again in the clip's own audio. This gives the exact AUTO start inside the clip.
   - The clip and its `stream_clip.json` go into the match's output folder.
5. **Tracking.** When a match is queued, the dashboard uses the clip and passes `--start-offset` from `stream_clip.json`, so the broadcast clock is not used at all. Calibration and approval work as before.

**Test stream:** FTC Meet @ Piper 1PM, event USMOKSKCWM7 (2h14m, 18 qualification matches).
- **Matches found:** all 18, each with both buzzers. No false matches, and all 18 labeled correctly. Q9 and Q11 match the clips of those matches used earlier.
- **Timing against the schedule:** every match is within ±31 s of its FTC Events time.
- **Timing against the video:** robots start moving 0.2-0.5 s after the computed AUTO start in every clip checked.
- **Speed:**
  - audio step (decode, detect, label): about 11 s for the whole stream; decoding was 29 s before the parallel decode;
  - full split including cutting all 18 clips from a local file: 33 s.
- **Load Event screen (browser):** stream field filled in, Load Event pressed. The split started by itself, all 18 matches were labeled and queueable, with no page errors.
- **End to end in the dashboard:** load the event live from FTC Events, split, reload, queue Q9. It ran field calibration on the clip with `--start-offset 9.45` and no download, and reached "awaiting approval" with high confidence.

**Download percentages:**
- **Process view:** every progress bar now shows its percentage next to it, including "Downloading match footage from YouTube", so you can see a download is really moving.
- **Split Stream status:** shows the percentage of the current download, e.g. "Downloading Qualification 3 (3 of 18) - 57%", for both the audio and each match section.
- **Tested** with a stand-in for yt-dlp that prints yt-dlp's progress lines and serves the local stream. The progress came through for every download, and the clips were cut with exact AUTO starts.

**Not tested here:** the YouTube download path, because YouTube blocks downloads from this build environment. It uses the same yt-dlp attempts (and cookies) as the normal match download.

**Limits:**
- It relies on the field buzzer being audible in the stream. A stream with the field audio muted finds no matches.
- If the processed clip was deleted after a successful run, processing that match again asks you to split the stream again.

### 13. New logo and disk cleanup (thirteenth commit)

**Logo:** the dashboard header shows the Juice 16236 + 9808 logo.
- It's stored in the repo as `dashboard_static/logo.png`: cropped, 96 px high, with a transparent background.
- Before, it loaded from the juicerobotics.org website, so it was missing without internet.

**Disk Space card** (Overview page, under Full Stream):
- **Shows** how much space the event uses, split into things that can safely be deleted:
  - **Full stream audio:** only needed to split the stream again.
  - **Match videos of finished matches:** downloads or stream clips.
  - **Processed overlay videos / frames of finished matches:** left unticked by default, since this is the tracking video you watch.
- **Free Up Space** deletes the ticked items after a confirmation.
- **Always kept:**
  - tracking results (CSV, logs, analytics, `done.json`);
  - any match that isn't finished;
  - any match whose job is queued, running or waiting for approval.
- **Endpoints:** `GET /api/event/disk-usage` and `POST /api/event/cleanup`.

**Checked** in the browser with test files:
- a finished match with its video, overlay frames and results;
- an unfinished match with its video;
- stream audio.

Cleanup deleted only the ticked stream audio and finished-match video (48 MB), and kept the results, the overlay frames and the unfinished match's video. No page errors, and the layout checked at desktop and phone width.

### 14. Longer stream clip margins (fourteenth commit)

**Problem:** stream clips started only 6 s before AUTO. A YouTube section download can begin a few seconds after the requested time, so a couple of seconds at the start of the match could be cut off.

**Fix:**
- Clips now start **16 s before AUTO** (10 s earlier than before) and end **10 s after the match** (was 6 s).
- The exact AUTO start inside each clip is still found from the clip's own buzzer, so tracking timing is unchanged.

**Checked** on all 18 matches of USMOKSKCWM7 (local stream):
- AUTO starts 16.2-19.5 s into every clip, found from the clip's own audio.
- Every clip has a full 10 s after the match end.
- Each clip is about 185 s.

### 15. No outside links except YouTube and FTC Events (fifteenth commit)

**Searched** every file in the repo for links to outside websites: URLs, web fonts, CDN scripts, CSS `url(...)`, and protocol-relative `//` links.

**Found and replaced:** the old juicerobotics.org logo was still loaded by the two browser tools, `tools/data_visualizer.html` (also embedded in the dashboard's match view) and `tools/manual_tracker.html`.
- They now use a local copy, `tools/logo.png` (the same new logo as the dashboard).
- Their logo sizing was updated for the wider logo.

**Left as is, because the software never loads them:**
- `http://www.w3.org/2000/svg`: an SVG namespace name, not a download;
- the README's demo video link (documentation only);
- the dashboard's own `http://127.0.0.1:port` address.

**Checked** in a browser, recording every network request:
- the dashboard, both tools through the dashboard, and the manual tracker opened straight from disk;
- **zero requests outside the local computer**, and the logo loads in all of them.

## Results on the example videos

| | Original / first version | Now |
|---|---|---|
| HIVE tips vs broadcast counter (q9 + q11 + Shanghai) | not available | 34 / 34 (also with automatic calibration) |
| q9 first auto volley (3 POLLEN into the red HIVE), checked frame by frame | shots drawn gray, not counted | 3 / 3 found, right robot |
| Field corners | placed by hand (bottom corners on the wall top, top corners too wide) | automatic at floor level, 2-4 corners visible |
| q9 impossible label jumps (faster than 90 in/s) | 421 | 56 |
| q11 impossible label jumps | 264 | 61 |
| Duplicate labels on one robot, BIOBUZZ | 4-8% of frames | under 1% |
| Duplicate labels on one robot, DECODE example | 46% of frames | 0% |
| DECODE example with the switches off | | identical to original |

## Switches in `autoscout/season.py`

All are on for BIOBUZZ. To reproduce the original DECODE tracker:
- set every switch below to off;
- set `FOOT_OFFSET_FRAC = 0`;
- set `INIT_STABLE_FRAMES = 1`;
- set `INIT_PARTIAL_AFTER_FRAMES` to a very large number;
- use the DECODE colors, with `CORNER_ZONE_SUPPRESSION = True`.

The switches:
- `PERSPECTIVE_ROBOT_SCALE`
- `AUTO_ROBOT_AREA`
- `PROTECT_VISIBLE_FROM_LOST`
- `STATIC_SUPPRESS_SPARES_LOST_TRACKS`
- `RESET_VELOCITY_ON_JUMP`
- `LOST_TRACK_RULES`
- `FOLLOW_TWO_ROBOT_MERGES`
- `HIVE_STATE_BACKGROUND`
- `AUTO_REID`
- `APPEARANCE_ID_CORRECTION`
- `REID_CORE_FRAC` (off by default)
- `FG_VERTICAL_CLOSE_PX`, `FRONT_RAIL_CLOSE_PX` (set to 0 to turn off)
- `MAX_BLOB_ROBOTS` (set to 0 to turn off)
- `MERGE_UNSEEN_RELEASE_S` (off by default)
- `AUTO_FIELD_CALIBRATION`, `FILTER_OFF_FIELD_PEOPLE`, `PERSON_MAX_ROBOT_HEIGHTS`, `GLITCH_FIELD_CHANGED_FRAC`
- `PROCESS_ROI`, `ROI_MARGIN_FRAC` (processing window; results do not depend on it on the examples)

## Known limitations

- **Labels are not guaranteed correct everywhere.** When similar-looking robots (for example the two silver robots in q9) are hidden together, appearance evidence is weak and a correction can be wrong. Labels after about 54 s in q9 could not be confirmed by eye at 640x360.
- **Shot counts are only spot-checked** (q9 first volley, a q11 stretch before a blue tip), not verified shot by shot.
  - Two or three balls flying up side by side at 640x360 are sometimes counted as one or two shots.
  - Shots launched from behind the HIVE are mostly invisible.
  - The red and blue HIVE boxes touch, so attribution near that border is unreliable.
  - Shooter credit goes to the robot nearest the first ball sighting.
- **The match start is set by hand** (`--start-offset`), with `tools/find_match_start.py` to help.
- **Robots are not linked to team numbers or alliances.**
- **Automatic calibration needs a check on overhead cameras** whose front field edge is under the overlay (the Edison DECODE stream): it reports low confidence there, and the approval step is where to fix it.
- **A referee standing in front of the camera** is mostly filtered out, but legs that overlap the field and don't reach the bottom of the picture can still show up as a foreground shape. Labels don't jump to it, and shapes bigger than 6 robots are ignored.
