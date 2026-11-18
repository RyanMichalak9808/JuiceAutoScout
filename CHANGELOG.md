# Changelog: BIOBUZZ (2026-27) adaptation

This file is a running summary of the changes made to AutoScout to adapt it from DECODE (2025-26) to BIOBUZZ (2026-27). For each change it records what it is, why it was needed, and the reasoning behind it. It also records how big the change is compared with the original repository (commit `4d1be90`) and how much more processing a match now needs.

## At a glance

- **New season:** POLLEN (yellow) shots into the red / blue HIVE, HIVE tip counting, and the BIOBUZZ field image.
- **New camera:** a low, audience-side event stream (640x360, blurry) instead of the high overhead DECODE World Championship broadcast.
- **Robot labels:** the main problem on the test videos was robot labels swapping or disappearing after collisions and behind the HIVE. Most of the tracker work went there.
- **Original behavior preserved:** every new behavior is behind a switch in `autoscout/season.py`. With the switches set back to their DECODE values, the tracker reproduces the original DECODE output exactly (0 differences over 6,128 rows of `examples/joosmatch.mp4`, rechecked after every round).

## Size of the change compared with the original repo

Measured with `git diff 4d1be90 HEAD`. Example videos and images are excluded.

| | Original | Now | Change |
|---|---|---|---|
| Code lines (`.py`, `.js`, `.html`) | 14,489 | 15,994 | **+1,688 / -183 lines, about 13% of the original code size touched; net +10%** |
| Original lines removed or replaced | | | 183 (1.3% of the original code) |
| Python lines | 8,025 | 9,530 | +19% |
| `autoscout/tracker.py` | 1,490 | 2,081 | +40% |
| `autoscout/shot.py` | 470 | 545 | +16% |
| `auto_scout.py` | 809 | 977 | +21% |
| New files | | | `autoscout/season.py`, `autoscout/hive.py`, `autoscout/match_start.py`, `tools/find_match_start.py`, `assets/biobuzz-field.png`, `CHANGELOG.md`, two example videos and their calibrations |

Most of the work is added code behind switches; very little original code was rewritten.

## Processing cost per match

The original and new versions were timed on the same machine (4 cores), one after the other, without `--debug-video`. Each was run twice.

| Video | Original | New | Wall time | CPU time (user + sys) |
|---|---|---|---|---|
| `piper_q9.mp4` (BIOBUZZ, 3 min, 640x360) | 66.2-66.4 s | 73.7-75.1 s | **+11 to +14%** | **+6 to +8%** (141 -> 151 s) |
| `joosmatch.mp4` (DECODE, 3.4 min) | 70.0 s | 75.7 s | +8% | +0% (152 -> 152 s) |

Where the extra time goes:

- HIVE tip detection, about 1 ms per frame.
- A second ball mask for shot detection.
- Per-position robot size and floor-contact projection (small homography calls).
- Robot appearance histograms for re-identification, plus the identity check.
- Learning the two HIVE backgrounds, once per match.

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

## Results on the example videos

| | Original / first version | Now |
|---|---|---|
| HIVE tips vs broadcast counter (q9 + q11) | not available | 13 / 13 |
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

## Known limitations

- **Labels are not guaranteed correct everywhere.** When similar-looking robots (for example the two silver robots in q9) are hidden together, appearance evidence is weak and a correction can be wrong. Labels after about 54 s in q9 could not be confirmed by eye at 640x360.
- **Shot counts are not verified shot by shot.**
  - Shots launched from behind the HIVE are mostly invisible.
  - The red and blue HIVE boxes touch, so attribution near that border is unreliable.
  - Shooter credit goes to the robot nearest the first ball sighting.
- **The match start is set by hand** (`--start-offset`), with `tools/find_match_start.py` to help.
- **Robots are not linked to team numbers or alliances.**
- **Calibration is manual for each camera setup.**
