![Project Header Image](assets/Project-REDACTED-Header.png)

# *FIRST* Tech Challenge Automatic Scouter (Project REDACTED)

https://github.com/user-attachments/assets/64045d5d-e12f-4a64-a7b1-fd415e46685a

![Analysis Screenahot](assets/example-analysis-1.webp)
<sub>*Shot Analysis*</sub>
![Analysis Screenahot](assets/example-analysis-2.webp)
<sub>*Movement Analysis*</sub>


## About AutoScout
`auto_scout.py` is the tracker entrypoint for FTC match videos. It orchestrates the tracking pipeline, projects robots into field coordinates, and exports the result for later analysis.

Current outputs are focused on robot motion:
- `robot_positions.csv` for per-timestamp robot poses and visibility
- `robot_positions.jlog` for compact binary robot poses and shot events
- `match_log.wpilog` for AdvantageScope playback
- optional annotated debug frames in `output/tracker_debug/`
- optional annotated debug video in `output/tracker_debug.mp4`

The CSV and JLOG pose logs include per-robot shot-event fields:

- `robot#_shot_result`
- `robot#_shot_x_in`
- `robot#_shot_y_in`
- `robot#_shot_goal`

These are populated on the frame where AutoScout resolves a shot as `made` or `missed`.

Output coordinates now use a center-origin field frame:

- `(0, 0)` is the center of the 144 inch x 144 inch field
- `x` increases to the right
- `y` increases downward in the projected field plane used by the tracker
- the field corners are approximately `(-72, -72)` to `(72, 72)` in inches

## Requirements
Install the runtime dependencies:

```bash
python3 -m pip install opencv-python numpy progress
python3 -m pip install scipy
```

`scipy` is technically optional, but the tracker uses it for the best blob-to-track assignment.

If you upgraded Python, reinstall optional tools like `yt-dlp` into the same interpreter you use to launch AutoScout:

```bash
python3 -m pip install -U yt-dlp
```

## Quickstart
Track a local video:

```bash
python3 auto_scout.py --no-download --video-path match.mp4 --corners field_corners.json
```

Track from YouTube:

```bash
python3 auto_scout.py "https://www.youtube.com/watch?v=..."
```

Start the local dashboard:

```bash
python3 dashboard_server.py --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765/` in a browser.

The dashboard adds:

- pause / resume: the automatic-processing button pauses new matches (optionally stopping running ones). You can shut the dashboard down and start it later: finished matches (`done.json`) and approved calibrations are kept, matches waiting for approval come back, and interrupted matches are processed again from the start

- FTC Events event loading by season plus event code
- qualification-match overview cards with alliance metadata, scores when available, and per-match status
- a hardware-aware global queue that can keep starting ready matches automatically
- local hardware inspection, runtime usage, and concurrency recommendations
- background tracker-job launching, stopping, live previews, stage checklists, terminal output, and embedded data visualization
- an in-browser field-corner calibration tool that can use uploaded files, queued YouTube clips, example videos, prior source videos, or still-cached match downloads
- yt-dlp cookies and extractor-args settings for both downloads and browser-playable YouTube resolution

### Full event streams

If an event only has one long stream, paste its YouTube link into **Full stream YouTube link (optional)** on the Load Event screen; the stream is split as soon as the event loads. (A local video path works too, and the split can be started later from **Full Stream** on the Overview page.) The field buzzers in the audio are used to find every match, the matches are labeled from the FTC Events schedule, and a clip with the exact AUTO start is cut for each one. The matches can then be processed like any other.

### Scouting

The **Analytics** page turns processed matches into scouting data: a sortable per-team table (shots made/missed, accuracy, AUTO shots, alliance tips, distance, red flags, next to EPA/OPR), a match-by-match view per team, and a menu for every processed match with its tables, tip timeline, heatmaps and CSV downloads. The tracker labels robots R0-R3; their teams are guessed from the HIVE they shoot into and can be confirmed per match (**Save robot teams**).

## CLI Flags
| Flag | Type | Default | Description |
| -------- | -------- | -------- | -------- |
| `--output-dir` | `string` | `./output` | Directory for CSV, JLOG, WPILOG, background image, and debug frames |
| `--start-offset` | `float` | `0.0` | Seconds to skip before the match timer starts |
| `--sample-rate` | `float` | `10.0` | Frames per second to track robots at (only used with `--no-every-frame`) |
| `--every-frame` / `--no-every-frame` | - | every frame | Track robots on every source frame, or at `--sample-rate`. Balls and touching robots are always followed every frame. Dashboard: Settings, Tracking |
| `--no-roi` | - | disabled | Process the whole frame instead of a window around the field. Dashboard: Settings, Tracking |
| `--debug` | - | disabled | Save annotated debug frames to `tracker_debug/` |
| `--debug-video` | - | disabled | Write annotated debug output to `tracker_debug.mp4` instead of `tracker_debug/` images. Implies `--debug` |
| `--debug-every` | `int` | `1` | Save one debug frame every `N` processed frames |
| `--no-download` | - | disabled | Use a local video instead of downloading from YouTube |
| `--video-path` | `string` | `None` | Path to the local match video. Requires `--no-download` |
| `--corners` | `string` | `None` | Path to `field_corners.json` from `tools/calibrate.py` |
| `--robot-init-positions` | `json` | `None` | Four starting field coordinates in inches, using center-origin coordinates |
| `--manual-reference-csv` | `string` | `None` | Manual `robot_positions`-style CSV or JLOG in the same center-origin coordinate system |

## Field Calibration
AutoScout can try to detect the field outline automatically, but manual calibration is more reliable.

Generate field corners from a local video:

```bash
python3 tools/calibrate.py path/to/match.mp4
```

That produces a `field_corners.json` file. Pass it to `auto_scout.py` with `--corners`. Recalibrate whenever the camera angle or crop changes.

## Season: BIOBUZZ (2026-2027)
AutoScout is configured for BIOBUZZ presented by RTX. Game-specific settings live in `autoscout/season.py`:

- HIVE tips are detected per alliance (`autoscout/hive.py`)
- shots are POLLEN (yellow balls) launched into the red / blue HIVE; NECTAR and FLOWERS are not tracked
- `robotN_shot_goal` is the HIVE color (`red` or `blue`)
- the tracker is set up for the typical low, audience-side BIOBUZZ camera: robot size varies with field position (`PERSPECTIVE_ROBOT_SCALE`) and positions use the robot's floor-contact point (`FOOT_OFFSET_FRAC`)
- robots that start hidden (for example behind the HIVE) are picked up when they appear
- robot labels are kept through contact and while hidden: no jumping across the field, hidden robots keep their label (hollow `R#?` marker in the debug video), and each robot's look is learned so swapped labels are corrected (see DOCUMENTATION.md, "Keeping robot labels")
- the visualizer uses `assets/biobuzz-field.png`, oriented as the camera sees the field (red on the left, far wall on top)

Field calibration is automatic: without `--corners`, AutoScout finds the floor-level field corners (even when only 2-3 are visible), the HIVE and FLOWER boxes and the overlay, and saves them as `field_corners_auto.json` with a preview image in the output folder. In the dashboard, every match first shows this calibration for approval (drag to fix, then Approve & Start Tracking). To check or fix a calibration by hand:

```bash
python3 tools/calibrate.py path/to/match.mp4
```

Press `A` to auto-detect, then drag the 4 corners to where the carpet meets the BOTTOM of the walls (they may lie in the gray border outside the picture). `H` draws the HIVE boxes, `F` a FLOWER box, `X` an area to ignore; `S` saves.

Calibration only (no tracking):

```bash
python3 auto_scout.py --no-download --video-path match.mp4 --calibrate-only --output-dir out
```

The broadcast clock and scoreboard are not reliable (the clock can start seconds after the robots), so the pipeline never reads them; `--auto-match-bounds` was built for the DECODE overlay. Set `--start-offset` to the moment robots first move or shoot. `tools/find_match_start.py` suggests candidates from the video and saves frames around each to confirm:

```bash
python3 tools/find_match_start.py match.mp4 --corners field_corners.json
```

Starting at the match start matters: field staff walking on the field before the match can otherwise be mistaken for robots.

```bash
python3 auto_scout.py --no-download --video-path match.mp4 --corners field_corners.json --start-offset 8
```

Example videos with calibrations (Piper Meet stream, 640x360):

```bash
python3 auto_scout.py --no-download --video-path examples/piper_q9.mp4 --corners examples/piper_q9_corners.json --start-offset 7.5 --debug-video --debug-every 3
python3 auto_scout.py --no-download --video-path examples/piper_q11.mp4 --corners examples/piper_q11_corners.json --start-offset 6 --debug-video --debug-every 3
```

```bash
python3 auto_scout.py --no-download --video-path examples/shanghai_q2_playoff5.mp4 --corners examples/shanghai_q2_playoff5_corners.json --start-offset 1.7 --debug-video --debug-every 3
```

HIVE tips are detected from the HIVE boxes and written to `hive_tips.csv`; on these three videos they match the broadcast tip counters exactly (34 tips).

## Repository Layout

- `auto_scout.py`
  Thin entrypoint that wires together the tracker pipeline, exports, and CLI.
- `dashboard_server.py`
  Starts the local dashboard web server.
- `autoscout/`
  Main Python package for the tracker internals.
  Includes `season.py`, `field_detect.py`, `hive.py`, `tracker.py`, `shot.py`, `runtime.py`, `geometry.py`, `models.py`, `helpers.py`, `wpilog.py`, `ftc_events.py`, `hardware.py`, and `dashboard_server.py`.
- `dashboard_static/`
  No-build dashboard front-end files served by `dashboard_server.py`.
- `util/`
  Shared helpers including `juice_log.py` and `jlog.js` for the compact `robot_positions.jlog` format.
- `tools/`
  Helper scripts and browser UIs: `calibrate.py`, `debug.py`, `manual_tracker.html`, `data_visualizer.html`, and `shot_visualizer.html`.
- `assets/`
  Static project images.
- `examples/`
  Example local videos for testing and manual workflows.
- `output/`
  Default generated outputs such as CSV, JLOG, WPILOG, background images, and debug media.
- `field_corners.json`
  A local calibration file in the repo root. You can also generate other corner JSON files and pass them with `--corners`.
- `README.md`, `DOCUMENTATION.md`, `AUTOSCOUT_PAPER.tex`
  Quickstart, full reference, and the paper-style writeup.

## Outputs
When a run finishes, the output directory typically contains:

- `robot_positions.csv`
  Flat table of timestamped robot positions, headings, visibility flags, and shot events. `x/y` are in inches from field center.
- `robot_positions.jlog`
  Compact binary JUICE LOG output for the same tracker table data, stored in a self-describing schema format with CRC-checked chunks so truncated writes keep earlier completed data readable.
- `hive_tips.csv`
  One row per HIVE tip (alliance, match time, approximate broadcast clock, phase, video time).
- `shot_events.csv`
  One row per resolved shot (robot, made/missed, HIVE, shooter position, times).
- `alliance_summary.csv`
  Per alliance: tips, auto tips, tips in the final 60 s / 15 s, first / last tip time, median time between tips (cycle cadence) and notes ("scores in AUTO", "stopped tipping in the final 60 s", "full-match HIVE cycler").
- `robot_summary.csv`
  Per robot track: distance driven, % of time moving, longest stall, field coverage, time on the left half, shots made / missed and red flags (long stall, never left one half, barely moved). Compare across a team's matches, after mapping R0-R3 to team numbers.
- `shot_heatmap.png`
  Shooter positions on the BIOBUZZ field map (made green, missed red).
- `resource_density.csv` / `resource_density.png`
  Loose POLLEN and red / blue NECTAR on the floor sampled every 2 s, drawn on the field map, with the share of POLLEN on each half early vs late in the match.
- `match_log.wpilog`
  WPILOG output using center-origin coordinates converted to meters with the viewer-facing axis remap `x' = y`, `y' = x`, and `heading' = π/2 - heading`.
- `median_background.jpg`
  The median background image used for subtraction.
- `tracker_debug/`
  Optional annotated frames showing contours, split centers, merge annotations, and robot IDs.
- `tracker_debug.mp4`
  Optional annotated debug video written when `--debug-video` is enabled.

When a job is launched from the dashboard, outputs are typically placed under:

- `<output_root>/<EVENT_CODE>/<match_slug>/`

where `output_root` defaults to `./output_dashboard`.

Dashboard YouTube jobs may also create a temporary `match_video.mp4` inside that match directory while the tracker is running. If the job completes successfully, the dashboard deletes that cached download after processing. If the job fails or is stopped before completion, the downloaded video is left in place so it can still be inspected, calibrated against, or reused manually.

## Debug Notes
- `--debug-every` is based on processed-frame spacing. The default `1` saves every processed frame.
- `--debug-video` writes only the sampled debug frames into the video, so playback speed is based on the debug sampling cadence rather than real-time source-video timing.
- Merge debug behavior can also be influenced by the top-level `SAVE_ALL_DEBUG_AROUND_MERGES` flag in `auto_scout.py`.
- Each debug run clears old `.jpg` files from `output/tracker_debug/` before writing new ones.

## AdvantageScope
To inspect the resulting WPILOG:

1. Open AdvantageScope.
2. Open `match_log.wpilog`.
3. Add a `2D Field` tab.
4. Set the field to the FTC season field you want.
5. Drag each `Robot#/Pose` entry into the field poses list.

## Contributing
See `CONTRIBUTING.md` for setup, validation, and pull request guidelines.

## License
This project is licensed under the MIT license, please see `LICENSE` for more information.
