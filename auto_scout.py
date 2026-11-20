import argparse
import base64
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

from autoscout.geometry import _field_center_to_corner_xy
from autoscout.helpers import (
    FieldDetector,
    _build_manual_reid_histograms,
    _draw_wire_cube,
    _open_debug_video_writer,
)
from autoscout import season
from autoscout import auto_paths
from autoscout.overrides import apply as apply_setting_overrides
from autoscout.match_timer import scan_match_bounds
from autoscout.runtime import _make_bar, _require, install_console_output
from autoscout import analytics
from autoscout.field_detect import FieldCalibrator, draw_calibration
from autoscout.hive import HiveTipDetector
from autoscout.shot import ShotDetector
from autoscout.tracker import RobotTracker
from autoscout.wpilog import WPILogWriter
from autoscout.ytdlp import (
    auth_help_message,
    build_download_attempts,
    find_yt_dlp_command,
    get_yt_dlp_command_warning,
    output_indicates_rate_limit,
    rate_limit_help_message,
    tls_help_message,
    yt_dlp_network_backoff_args,
)
from util.juice_log import (
    CSV_COLUMNS,
    ROBOT_POSE_SCHEMA,
    JuiceLogWriter,
    csv_row_to_list,
)


SAVE_ALL_DEBUG_AROUND_MERGES = False   # save every debug frame while robots are merged (for debugging label swaps)
PROCESS_EVERY_SOURCE_FRAME = True
SHOT_DETECTOR_FPS = 30.0   # frame rate the shot detector's per-frame tuning is for
DASHBOARD_EVENT_PREFIX = "@@AUTOSCOUT_DASHBOARD@@"
apply_setting_overrides("auto_scout", globals())   # values changed in the dashboard Settings tab


install_console_output()


def _dashboard_mode_enabled():
    return os.environ.get("AUTOSCOUT_DASHBOARD_MODE") == "1"


def _dashboard_preview_interval():
    raw = os.environ.get("AUTOSCOUT_DASHBOARD_PREVIEW_EVERY", "5").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


def _emit_dashboard_event(kind, **payload):
    if not _dashboard_mode_enabled():
        return
    event = {"kind": kind}
    event.update(payload)
    print(DASHBOARD_EVENT_PREFIX + json.dumps(event, separators=(",", ":")), flush=True)


def _emit_dashboard_stage(stage, status, detail=""):
    _emit_dashboard_event("stage", stage=stage, status=status, detail=detail)


def _emit_dashboard_stage_progress(stage, detail, current, total):
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    _emit_dashboard_event(
        "stage",
        stage=stage,
        status="in_progress",
        detail=detail,
        progress_current=current,
        progress_total=total,
        progress_fraction=(current / total),
    )


def _emit_dashboard_preview(cv2, frame, frame_num, match_time_s):
    if not _dashboard_mode_enabled():
        return
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 70],
    )
    if not ok:
        return
    _emit_dashboard_event(
        "preview",
        frame_num=int(frame_num),
        match_time_s=round(float(match_time_s), 2),
        jpeg_b64=base64.b64encode(encoded.tobytes()).decode("ascii"),
    )


def _manual_corners_file_to_tracker_order(manual_corners_px):
    if not manual_corners_px or len(manual_corners_px) != 4:
        return manual_corners_px
    # Saved calibration files are persisted as [BL, BR, TR, TL].
    # The tracker homography expects [TL, TR, BR, BL].
    bl, br, tr, tl = manual_corners_px
    return [tl, tr, br, bl]


def process_match(
    video_path,
    output_dir,
    start_offset_sec=0.0,
    sample_rate_fps=10.0,
    process_every_frame=None,
    auto_only=False,
    debug=False,
    debug_video=False,
    debug_every_n=1,
    debug_enable_hitboxes=False,
    manual_corners_px=None,
    robot_init_positions=None,
    manual_reference_csv=None,
    auto_match_bounds=False,
    calibration_extras=None,
):
    cv2 = _require("cv2", "opencv-python")
    np = _require("numpy")
    dashboard_stream_previews = _dashboard_mode_enabled() and debug and not debug_video

    os.makedirs(output_dir, exist_ok=True)

    _emit_dashboard_stage("Loading Match Footage", "in_progress", "Opening match footage")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _emit_dashboard_stage("Loading Match Footage", "failed", "Unable to open match footage")
        print("[ERROR] Cannot open: {}".format(video_path))
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] Video: {:.1f} fps, {} frames, {:.1f}s".format(
        video_fps, total_frames, total_frames / video_fps))

    start_frame = int(start_offset_sec * video_fps)
    stop_frame = total_frames - 1
    if auto_match_bounds:
        timer_scan = scan_match_bounds(
            video_path,
            cv2,
            np,
            progress_callback=lambda current, total: _emit_dashboard_stage_progress(
                "Loading Match Footage",
                "Scanning broadcast timer for match bounds",
                current,
                total,
            ),
        )
        auto_start_frame = timer_scan.get("start_frame")
        auto_stop_frame = timer_scan.get("stop_frame")
        if auto_start_frame is not None:
            start_frame = max(0, int(auto_start_frame))
            start_offset_sec = start_frame / video_fps
            print("[INFO] Auto-detected match start at {:.2f}s (frame {}).".format(
                start_offset_sec,
                start_frame,
            ))
        else:
            start_offset_sec = 0.0
            start_frame = 0
            print("[WARN] Could not auto-detect the match start timer. Starting from frame 0.")
        if auto_stop_frame is not None:
            stop_frame = int(auto_stop_frame)
            stop_time_s = stop_frame / video_fps
            print("[INFO] Auto-detected match end at {:.2f}s (frame {}).".format(
                stop_time_s,
                stop_frame,
            ))
        else:
            print("[WARN] Could not auto-detect the end of match timer. Processing until the video ends.")
    else:
        print("[INFO] Using explicit start offset of {:.2f}s (frame {}).".format(
            start_offset_sec,
            start_frame,
        ))

    if auto_only:
        # scouting AUTO paths only: stop shortly after AUTO ends
        auto_stop = start_frame + int(round((season.AUTO_END_S + 2.0) * video_fps))
        if auto_stop < stop_frame:
            stop_frame = auto_stop
        print("[INFO] AUTO only: tracking stops at {:.1f}s match time.".format(season.AUTO_END_S + 2.0))
    frame_step = max(1, int(round(video_fps / sample_rate_fps)))
    if PROCESS_EVERY_SOURCE_FRAME if process_every_frame is None else process_every_frame:
        frame_step = 1
    print("[INFO] Processing every {} frames (~{:.1f} fps output)".format(
        frame_step, video_fps / frame_step))
    # Debug frames are drawn on tracked frames only, so the debug cadence is a
    # whole number of tracking steps; the debug video is written at the
    # matching rate and plays back in real time at any tracking rate.
    debug_every_n = frame_step * int(math.ceil(max(1, int(debug_every_n)) / frame_step))
    _emit_dashboard_stage("Loading Match Footage", "completed", "Match footage is ready")
    _emit_dashboard_stage("Calibrating Trackers", "in_progress", "Preparing tracker calibration")
    if debug:
        if debug_video:
            print("[INFO] Writing debug video from frames sampled about every {} source frames".format(
                debug_every_n))
        else:
            print("[INFO] Saving debug frame about every {} source frames".format(debug_every_n))
        if debug_enable_hitboxes:
            print("[INFO] Debug robot wireframe hitboxes enabled.")

    csv_path = os.path.join(output_dir, "robot_positions.csv")
    jlog_path = os.path.join(output_dir, "robot_positions.jlog")
    wpilog_path = os.path.join(output_dir, "match_log.wpilog")
    debug_dir = os.path.join(output_dir, "tracker_debug") if (debug and not debug_video and not dashboard_stream_previews) else None
    debug_video_path = os.path.join(output_dir, "tracker_debug.mp4") if (debug and debug_video) else None

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for name in os.listdir(debug_dir):
            if name.endswith(".jpg"):
                try:
                    os.remove(os.path.join(debug_dir, name))
                except OSError:
                    pass
    if debug_video_path and os.path.exists(debug_video_path):
        try:
            os.remove(debug_video_path)
        except OSError:
            pass
    avi_fallback_path = os.path.join(output_dir, "tracker_debug.avi")
    if debug_video_path and os.path.exists(avi_fallback_path):
        try:
            os.remove(avi_fallback_path)
        except OSError:
            pass

    log = WPILogWriter(wpilog_path)
    pose_eids = [log.start_entry("Robot{}/Pose".format(i), "double[]") for i in range(4)]
    vis_eids = [log.start_entry("Robot{}/Visible".format(i), "boolean") for i in range(4)]

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_COLUMNS)
    jlog_writer = JuiceLogWriter(jlog_path, schema=ROBOT_POSE_SCHEMA)

    tracker = RobotTracker(cv2, np)
    shot_detector = ShotDetector(cv2, np)
    hive_detector = HiveTipDetector(cv2, np)
    shot_log = []   # rows for shot_events.csv
    tip_log = []    # rows for hive_tips.csv
    speed_hist = [[] for _ in range(4)]  # per robot: (match_time_s, x_in, y_in) for debug speed arrows
    field_detector = FieldDetector(cv2, np)

    ordered = None
    H_2d = None
    H_inv = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, sample_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame.")
        sys.exit(1)
    frame_shape = sample_frame.shape
    debug_writer = None
    if debug_video_path:
        debug_h, debug_w = frame_shape[:2]
        debug_fps = max(1.0, video_fps / max(1, debug_every_n))
        debug_writer, opened_path_or_attempts, debug_codec = _open_debug_video_writer(
            cv2,
            output_dir,
            (debug_w, debug_h),
            debug_fps,
        )
        if debug_writer is None:
            print("[WARN] Could not open debug video writer. Tried:")
            for attempt in opened_path_or_attempts:
                print("  {}".format(attempt))
        else:
            debug_video_path = opened_path_or_attempts
            print("[INFO] Debug video: {} [{} @ {:.2f} fps]".format(
                debug_video_path, debug_codec, debug_fps))

    auto_calibration_failed = False
    if not manual_corners_px and season.AUTO_FIELD_CALIBRATION:
        # No calibration given: detect the field (corners at floor level, HIVE,
        # FLOWER and overlay boxes) from the video itself.
        _emit_dashboard_stage("Calibrating Trackers", "in_progress", "Detecting field corners")
        auto_cal = auto_calibrate_video(cv2, np, video_path, output_dir,
                                        start_s=start_frame / video_fps,
                                        end_s=stop_frame / video_fps)
        if auto_cal is not None:
            manual_corners_px = auto_cal["corners_px"]
            calibration_extras = dict(calibration_extras or {})
            for key in ("hive_boxes_px", "exclude_px", "flower_boxes_px"):
                if key in auto_cal and key not in calibration_extras:
                    calibration_extras[key] = auto_cal[key]
            print("[INFO] Using automatic field calibration ({}, {} confidence). Check {} .".format(
                auto_cal["auto"]["method"], auto_cal["auto"]["confidence"],
                os.path.join(output_dir, "field_corners_auto_preview.jpg")))
        else:
            auto_calibration_failed = True
            print("[WARN] Falling back to the per-frame field outline detector.")

    if manual_corners_px:
        ordered = np.array(_manual_corners_file_to_tracker_order(manual_corners_px), dtype=np.float32)
        dst2d = np.array([[0, 0], [144, 0], [144, 144], [0, 144]], dtype=np.float32)
        H_2d, _ = cv2.findHomography(ordered, dst2d)
        H_inv = np.linalg.inv(H_2d)
        tracker.setup(video_path, ordered, frame_shape, calibration_extras)
        shot_detector.setup(tracker)
        hive_detector.setup(tracker)
        print("[INFO] Using manual field corners.")
        _emit_dashboard_stage("Calibrating Trackers", "completed", "Using manual field corners")
        _emit_dashboard_stage("Tracking Match", "in_progress", "Tracking robots across the match")
        if tracker._bg is not None:
            cv2.imwrite(os.path.join(output_dir, "median_background.jpg"), tracker._bg)

    if robot_init_positions:
        if H_inv is None:
            print("[ERROR] --robot-init-positions requires field corners to be known.")
            sys.exit(1)
        if len(robot_init_positions) != 4:
            print("[ERROR] --robot-init-positions must provide exactly 4 [x,y] pairs.")
            sys.exit(1)
        lineup = []
        for idx, item in enumerate(robot_init_positions):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                print("[ERROR] Robot init entry {} is invalid: {}".format(idx, item))
                sys.exit(1)
            fx_center = float(item[0])
            fy_center = float(item[1])
            fx, fy = _field_center_to_corner_xy(fx_center, fy_center)
            ipx, ipy = tracker.field_to_image(fx, fy)
            lineup.append((fx, fy, ipx, ipy))
        tracker._initialize_tracks_from_lineup(lineup, "manual --robot-init-positions")

    if manual_reference_csv:
        if H_inv is None:
            print("[ERROR] --manual-reference-csv requires field corners to be known.")
            sys.exit(1)
        refs = _build_manual_reid_histograms(
            tracker,
            cap,
            manual_reference_csv,
            H_inv,
            video_fps,
        )
        if refs:
            tracker.set_reid_reference_histograms(refs)
            print("[INFO] Manual re-ID appearance model enabled.")

    frame_num = start_frame
    processed = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame if start_frame > 0 else 0)

    debug_colors = [(220, 160, 0), (0, 220, 255), (0, 0, 220), (0, 120, 255)]
    end_frame_exclusive = min(total_frames, stop_frame + 1)
    frames_to_process = max(1, (end_frame_exclusive - start_frame + frame_step - 1) // frame_step)
    bar = _make_bar("Processing frames", frames_to_process)
    merge_debug_hold = 0
    dense_merge_hold = 0
    next_debug_source_frame = start_frame
    dashboard_preview_every = max(debug_every_n, _dashboard_preview_interval()) if dashboard_stream_previews else max(1, debug_every_n)
    last_poses = None
    resource_sampler = analytics.ResourceSampler(cv2, np)
    glitch_frames = 0
    pending_shot_events = []
    # The shot detector is tuned for ~30 fps (its gates, gravity and speeds are
    # per frame). On 60 fps video it gets every 2nd frame, on a ~30 fps index,
    # so the same tuning applies; event frame numbers are mapped back.
    ball_step = max(1, int(round(video_fps / SHOT_DETECTOR_FPS)))
    shot_detector._fps = video_fps / ball_step
    if ball_step > 1:
        print("[INFO] Shot detector runs on every {} frames (~{:.0f} fps).".format(ball_step, video_fps / ball_step))

    def shot_update(fnum, t, poses):
        if (fnum - start_frame) % ball_step:
            return []
        events = shot_detector.update(
            (fnum - start_frame) // ball_step, t, poses, tracker,
            hive_moving=lambda color: hive_detector.is_moving(color, t))
        for event in events:
            event.frame_num = start_frame + event.frame_num * ball_step
        return events
    calibration_completed = H_inv is not None
    tracking_started = H_inv is not None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_num = frame_num
        if current_frame_num > stop_frame:
            break
        match_time_s = current_frame_num / video_fps - start_offset_sec
        timestamp_us = max(0, int(match_time_s * 1_000_000))
        frame_num += 1

        process_dense = dense_merge_hold > 0 or bool(tracker._merge_groups)
        if not process_dense and (current_frame_num - start_frame) % frame_step != 0:
            # Robot tracking skips this frame, but balls are followed on every
            # frame: fast vertical shots cross the view in a few frames.
            if H_inv is not None and last_poses is not None \
                    and (current_frame_num - start_frame) % ball_step == 0 \
                    and not tracker.is_glitch_frame(frame):
                tracker.ball_mask_only(frame)
                pending_shot_events.extend(shot_update(current_frame_num, match_time_s, None))
            continue

        bar.next()
        if processed == 0 or processed % 30 == 0 or processed + 1 >= frames_to_process:
            _emit_dashboard_stage_progress(
                "Tracking Match",
                "Tracking robots across the match",
                processed + 1,
                frames_to_process,
            )

        if ordered is None and (not season.AUTO_FIELD_CALIBRATION or auto_calibration_failed):
            result = field_detector.detect_field(frame)
            if result is not None:
                ordered, H_2d = result
                H_inv = np.linalg.inv(H_2d)
                tracker.setup(video_path, ordered, frame.shape, calibration_extras)
                shot_detector.setup(tracker)
                hive_detector.setup(tracker)
                print("\n  [t={:.1f}s] Field auto-detected.".format(match_time_s))
                calibration_completed = True
                _emit_dashboard_stage("Calibrating Trackers", "completed", "Field auto-detected")
                if not tracking_started:
                    tracking_started = True
                    _emit_dashboard_stage("Tracking Match", "in_progress", "Tracking robots across the match")
                if tracker._bg is not None:
                    cv2.imwrite(os.path.join(output_dir, "median_background.jpg"), tracker._bg)

        if tracker._bg is not None and tracker.is_glitch_frame(frame):
            glitch_frames += 1
            continue
        tip_events = hive_detector.update(frame, current_frame_num, match_time_s)
        tracker.set_hive_state(hive_detector.state(match_time_s))
        poses = tracker.update(frame, time_s=match_time_s)
        last_poses = poses
        if 0.0 <= match_time_s <= season.MATCH_END_S:
            resource_sampler.maybe_sample(frame, match_time_s, tracker)
        shot_events = pending_shot_events + shot_update(current_frame_num, match_time_s, poses)
        pending_shot_events = []
        for tip in tip_events:
            tip_log.append(_tip_row(tip, video_fps))
            print("[INFO] HIVE tip: {} #{} at match t={:.1f}s (video {:.1f}s, {})".format(
                tip.alliance, tip.tip_number, tip.match_time_s,
                tip.frame_num / video_fps, _match_phase(tip.match_time_s)))
        for event in shot_events:
            shot_log.append(_shot_row(event, video_fps))
        for robot_id, pose in enumerate(poses):
            hist = speed_hist[robot_id]
            if pose.visible:
                hist.append((match_time_s, pose.x_in, pose.y_in))
                while hist and match_time_s - hist[0][0] > 0.5:
                    hist.pop(0)
            else:
                hist.clear()
        merge_active = bool(tracker._merge_groups)
        recent_merge = any(v > 0 for v in tracker._merge_recent)
        if merge_active or recent_merge:
            dense_merge_hold = tracker.MERGE_DEBUG_HOLD
        elif dense_merge_hold > 0:
            dense_merge_hold -= 1
        if merge_active:
            merge_debug_hold = tracker.MERGE_DEBUG_HOLD
        elif merge_debug_hold > 0:
            merge_debug_hold -= 1

        for i, pose in enumerate(poses):
            log.write_pose2d(
                pose_eids[i],
                timestamp_us,
                pose.wpilog_x_m,
                pose.wpilog_y_m,
                pose.wpilog_heading_rad,
            )
            log.write_boolean(vis_eids[i], timestamp_us, pose.visible)

        row = {"timestamp_s": "{:.4f}".format(match_time_s)}
        for robot_id, pose in enumerate(poses):
            row["robot{}_x_in".format(robot_id)] = "{:.2f}".format(pose.x_center_in)
            row["robot{}_y_in".format(robot_id)] = "{:.2f}".format(pose.y_center_in)
            row["robot{}_heading_rad".format(robot_id)] = "{:.4f}".format(pose.heading)
            row["robot{}_visible".format(robot_id)] = "1" if pose.visible else "0"
        shot_cells = [["", "", "", ""] for _ in range(4)]
        for event in shot_events:
            if 0 <= event.shooter_id < 4:
                shot_cells[event.shooter_id] = [
                    event.result,
                    "{:.2f}".format(event.shot_x_in),
                    "{:.2f}".format(event.shot_y_in),
                    event.goal_color,
                ]
        for robot_id, cells in enumerate(shot_cells):
            row["robot{}_shot_result".format(robot_id)] = cells[0]
            row["robot{}_shot_x_in".format(robot_id)] = cells[1]
            row["robot{}_shot_y_in".format(robot_id)] = cells[2]
            row["robot{}_shot_goal".format(robot_id)] = cells[3]
        csv_writer.writerow(csv_row_to_list(row))
        jlog_writer.append_row(row)

        if debug and H_inv is not None and (dashboard_stream_previews or debug_dir or debug_writer is not None):
            dbg = frame.copy()
            fg_debug = None
            if ordered is not None:
                cv2.polylines(dbg, [ordered.astype(np.int32).reshape(-1, 1, 2)],
                              True, (0, 200, 0), 2)
                shot_detector.draw_debug(dbg)
                hive_detector.draw_debug(dbg)
            for ex_poly in tracker.exclude_polys:
                cv2.polylines(dbg, [ex_poly], True, (128, 128, 128), 1, cv2.LINE_AA)

            if tracker._bg is not None:
                fg_debug = tracker._foreground_mask(frame)
                all_cnts, split_centers = tracker._foreground_contours(frame)
                if tracker._last_ball_mask is not None:
                    ball_cnts, _ = cv2.findContours(
                        tracker._last_ball_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    for contour in ball_cnts:
                        if cv2.contourArea(contour) >= 2:
                            cv2.drawContours(dbg, [contour], -1, (255, 0, 255), 1)
                for contour in all_cnts:
                    area = cv2.contourArea(contour)
                    mask = np.zeros(tracker._field_mask.shape, np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                    is_robot = dist.max() >= tracker.MIN_RADIUS_PX
                    color = (0, 255, 255) if is_robot else (160, 160, 160)
                    cv2.drawContours(dbg, [contour], -1, color, 2 if is_robot else 1)
                    if area >= tracker.BLOB_MIN:
                        moments = cv2.moments(contour)
                        if moments["m00"] > 0:
                            bx = int(moments["m10"] / moments["m00"])
                            by = int(moments["m01"] / moments["m00"])
                            cv2.putText(dbg, "{:.0f}".format(area),
                                        (bx - 15, by + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                for sx, sy in split_centers[:4]:
                    cv2.drawMarker(dbg, (sx, sy), (0, 255, 255),
                                   cv2.MARKER_CROSS, 16, 2)
                # Robot-sized shapes with no visible track on them.
                track_pts = [tracker._pos_px[t] for t in range(4)
                             if tracker._pos_px[t] is not None and poses[t].visible]
                for contour in all_cnts:
                    if any(cv2.pointPolygonTest(contour, (float(px), float(py)), True)
                           >= -0.5 * tracker._contour_robot_px(contour)
                           for px, py in track_pts):
                        continue
                    bx, by, bw, _bh = cv2.boundingRect(contour)
                    cv2.putText(dbg, "no track", (bx, max(10, by - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            fh_d, fw_d = frame.shape[:2]
            cv2.circle(dbg, (fw_d - 40, 40),
                       int(tracker._robot_max_px / 2), (80, 80, 80), 1)
            cv2.putText(dbg, "18in", (fw_d - 74, 57),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)

            for merge_group in tracker._merge_groups.values():
                pos_list = [tracker._pos_px[tid] for tid in merge_group.track_ids
                            if tracker._pos_px[tid] is not None]
                if pos_list:
                    mcx = int(sum(point[0] for point in pos_list) / len(pos_list))
                    mcy = int(sum(point[1] for point in pos_list) / len(pos_list))
                    ax, ay = merge_group.entry_axis
                    ex = int(mcx + ax * 50)
                    ey = int(mcy + ay * 50)
                    if len(merge_group.track_ids) == 2:
                        color = (0, 80, 255) if merge_group.crossed else (0, 220, 100)
                        label = "+".join("R{}".format(tid) for tid in merge_group.track_ids)
                        label += " CROSSED" if merge_group.crossed else " ok"
                    else:
                        color = (255, 160, 0)
                        entry_s = "".join(str(tid) for tid in merge_group.entry_order)
                        current_s = "".join(str(tid) for tid in merge_group.current_order)
                        label = "+".join("R{}".format(tid) for tid in merge_group.track_ids)
                        label += " [{}->{}]".format(entry_s, current_s)
                    cv2.arrowedLine(dbg, (mcx, mcy), (ex, ey), color, 2)
                    cv2.putText(dbg, label, (mcx + 6, mcy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            n_drawn = 0
            fh, fw = frame.shape[:2]
            # Hidden robots: hollow marker with "R#?" at the last known spot.
            for i, pose in enumerate(poses):
                if pose.visible or tracker._pos[i] is None:
                    continue
                hx, hy = tracker._pos[i]
                hp = cv2.perspectiveTransform(np.array([[[hx, hy]]], dtype=np.float32), H_inv)[0][0]
                hix, hiy = int(hp[0]), int(hp[1])
                if 0 <= hix < fw and 0 <= hiy < fh:
                    cv2.circle(dbg, (hix, hiy), 14, debug_colors[i], 2)
                    cv2.putText(dbg, "R{}?".format(i), (hix - 12, hiy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, debug_colors[i], 2)
            for i, pose in enumerate(poses):
                if not pose.visible:
                    continue
                pt = np.array([[[pose.x_in, pose.y_in]]], dtype=np.float32)
                img = cv2.perspectiveTransform(pt, H_inv)[0][0]
                ix, iy = int(img[0]), int(img[1])
                if not (0 <= ix < fw and 0 <= iy < fh):
                    continue
                color = debug_colors[i]
                cv2.circle(dbg, (ix, iy), 16, color, -1)
                cv2.circle(dbg, (ix, iy), 19, (255, 255, 255), 2)
                if debug_enable_hitboxes and fg_debug is not None:
                    _draw_wire_cube(cv2, np, dbg, frame, fg_debug,
                                    tracker, H_inv, pose, (ix, iy), color)
                cv2.putText(dbg, "R{}".format(i), (ix - 10, iy - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Speed arrow from the robot's actual motion over the last
                # 0.5 s: direction in the image, length grows with speed.
                hist = speed_hist[i]
                if len(hist) >= 2 and hist[-1][0] - hist[0][0] > 0.15:
                    t0, x0, y0 = hist[0]
                    t1, x1, y1 = hist[-1]
                    speed_in_s = math.hypot(x1 - x0, y1 - y0) / max(t1 - t0, 1e-6)
                    if speed_in_s >= 3.0:
                        p0 = cv2.perspectiveTransform(np.array([[[x0, y0]]], dtype=np.float32), H_inv)[0][0]
                        p1 = cv2.perspectiveTransform(np.array([[[x1, y1]]], dtype=np.float32), H_inv)[0][0]
                        dxp, dyp = float(p1[0] - p0[0]), float(p1[1] - p0[1])
                        norm = math.hypot(dxp, dyp)
                        if norm > 1e-6:
                            length = min(70.0, 14.0 + 0.6 * speed_in_s)
                            end = (int(round(ix + dxp / norm * length)), int(round(iy + dyp / norm * length)))
                            cv2.arrowedLine(dbg, (ix, iy), end, (255, 255, 255), 2, tipLength=0.25)
                            cv2.putText(dbg, "{:.0f} in/s".format(speed_in_s), (end[0] + 3, end[1] + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                n_drawn += 1

            n_merged = sum(len(merge_group.track_ids) for merge_group in tracker._merge_groups.values())
            label = ("init" if not tracker._initialized else
                     "{}/4 ({} merged)".format(n_drawn, n_merged) if n_merged else
                     "{}/4".format(n_drawn))
            cv2.putText(dbg, "t={:.2f}s f={}  [{}]".format(
                        match_time_s, current_frame_num, label),
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            base_debug_interval = dashboard_preview_every
            save_debug = (current_frame_num >= next_debug_source_frame
                          or (SAVE_ALL_DEBUG_AROUND_MERGES and
                              (merge_active or merge_debug_hold > 0)))
            if save_debug:
                if debug_writer is not None:
                    debug_writer.write(dbg)
                elif dashboard_stream_previews:
                    _emit_dashboard_preview(cv2, dbg, current_frame_num, match_time_s)
                elif debug_dir is not None:
                    cv2.imwrite(os.path.join(debug_dir,
                                "frame_{:06d}.jpg".format(current_frame_num)), dbg)
                if current_frame_num >= next_debug_source_frame:
                    while current_frame_num >= next_debug_source_frame:
                        next_debug_source_frame += base_debug_interval

        for event in shot_events:
            print("[INFO] Shot {} by {} at ({:.1f},{:.1f}) goal={}".format(
                event.result,
                "R{}".format(event.shooter_id) if event.shooter_id >= 0 else "unknown robot",
                event.shot_x_in,
                event.shot_y_in,
                event.goal_color or "unknown",
            ))

        processed += 1

    if not calibration_completed:
        _emit_dashboard_stage("Calibrating Trackers", "failed", "Field calibration could not be established")
    elif tracking_started:
        _emit_dashboard_stage("Tracking Match", "completed", "Tracking loop finished")
    _emit_dashboard_stage("Cleaning Up", "in_progress", "Finalizing outputs")
    bar.finish()
    if debug_writer is not None:
        debug_writer.release()
    log.close()
    csv_file.close()
    jlog_writer.close()
    cap.release()
    if glitch_frames:
        print("[INFO] Skipped {} frame(s) with a camera cut/flash (most of the field changed at once).".format(glitch_frames))
    tips_path = os.path.join(output_dir, "hive_tips.csv")
    shots_path = os.path.join(output_dir, "shot_events.csv")
    _write_rows(tips_path, TIP_LOG_COLUMNS, tip_log)
    _write_rows(shots_path, SHOT_LOG_COLUMNS, shot_log)
    print("\n[DONE] {} frames processed.".format(processed))
    print("  CSV:    {}".format(csv_path))
    print("  JLOG:   {}".format(jlog_path))
    print("  WPILOG: {}".format(wpilog_path))
    print("  TIPS:   {}".format(tips_path))
    print("  SHOTS:  {}".format(shots_path))
    _print_event_summary(tip_log, shot_log)
    try:
        with open(csv_path, newline="") as handle:
            auto = auto_paths.write(cv2, np, output_dir, list(csv.DictReader(handle)))
        print("  AUTO paths: auto_paths.json, auto_paths.csv, auto_paths.png")
        for r in auto:
            if r["start"]:
                print("    {} start ({:.0f}, {:.0f}) -> end ({:.0f}, {:.0f}) in | {:.0f} in driven | seen {}% of AUTO".format(
                    r["robot"], r["start"][0], r["start"][1], r["end"][0], r["end"][1],
                    r["distance_in"], r["visible_pct"]))
    except Exception as exc:  # AUTO paths must never fail the run
        print("[WARN] AUTO paths not written: {}".format(exc))
    with open(os.path.join(output_dir, "done.json"), "w") as handle:
        json.dump({"finished_at": time.time(), "frames_processed": processed,
                   "video": os.path.abspath(video_path), "auto_only": bool(auto_only)}, handle, indent=2)
    try:
        alliances, robots = analytics.write_all(cv2, np, output_dir, csv_path, tip_log, shot_log,
                                                resource_sampler.samples)
        print("  Analytics: alliance_summary.csv, robot_summary.csv, shot_heatmap.png, resource_density.png")
        for a in alliances:
            print("    {:<5} tips {} (auto {}, final 60 s {}) | median {} s between tips | {}".format(
                a["alliance"], a["tips"], a["auto_tips"], a["endgame_tips_final_60s"],
                a["median_s_between_tips"] or "-", a["notes"] or "-"))
        for r in robots:
            if r["red_flags"]:
                print("    {} red flags: {}".format(r["robot"], r["red_flags"]))
    except Exception as exc:  # analytics must never break a finished run
        print("[WARN] Analytics failed: {}".format(exc))
    if debug:
        debug_location = debug_video_path if debug_video_path else (debug_dir or "dashboard stream")
        print("  Debug:  {}".format(debug_location))
    _emit_dashboard_stage("Cleaning Up", "completed", "Outputs are ready")
    _print_instructions()


TIP_LOG_COLUMNS = [
    "alliance", "tip_number", "match_time_s", "match_clock", "phase",
    "video_time_s", "frame", "level_before", "level_after",
]
SHOT_LOG_COLUMNS = [
    "match_time_s", "match_clock", "phase", "video_time_s", "frame",
    "robot", "result", "hive", "shooter_x_in", "shooter_y_in",
]


def _match_phase(match_time_s):
    if match_time_s < 0:
        return "pre_match"
    if match_time_s < season.AUTO_END_S:
        return "auto"
    if match_time_s < season.TELEOP_START_S:
        return "transition"
    if match_time_s <= season.MATCH_END_S:
        return "teleop"
    return "after_end"


def _match_clock(match_time_s):
    """Approximate broadcast clock: counts 2:30 -> 2:00 in auto, holds at
    2:00 during the transition, then counts 2:00 -> 0:00 in teleop."""
    if match_time_s < season.AUTO_END_S:
        remaining = (season.MATCH_END_S - season.TELEOP_START_S
                     + season.AUTO_END_S - max(0.0, match_time_s))
    elif match_time_s < season.TELEOP_START_S:
        remaining = season.MATCH_END_S - season.TELEOP_START_S
    else:
        remaining = max(0.0, season.MATCH_END_S - match_time_s)
    remaining = int(math.ceil(remaining - 1e-6))
    return "{}:{:02d}".format(remaining // 60, remaining % 60)


def _tip_row(tip, video_fps):
    return {
        "alliance": tip.alliance,
        "tip_number": tip.tip_number,
        "match_time_s": "{:.2f}".format(tip.match_time_s),
        "match_clock": _match_clock(tip.match_time_s),
        "phase": _match_phase(tip.match_time_s),
        "video_time_s": "{:.2f}".format(tip.frame_num / video_fps),
        "frame": tip.frame_num,
        "level_before": "{:.3f}".format(tip.level_before),
        "level_after": "{:.3f}".format(tip.level_after),
    }


def _shot_row(event, video_fps):
    return {
        "match_time_s": "{:.2f}".format(event.timestamp_s),
        "match_clock": _match_clock(event.timestamp_s),
        "phase": _match_phase(event.timestamp_s),
        "video_time_s": "{:.2f}".format(event.frame_num / video_fps),
        "frame": event.frame_num,
        "robot": "R{}".format(event.shooter_id) if event.shooter_id >= 0 else "unknown",
        "result": event.result,
        "hive": event.goal_color or "unknown",
        "shooter_x_in": "{:.1f}".format(event.shot_x_in) if event.shot_x_in == event.shot_x_in else "",
        "shooter_y_in": "{:.1f}".format(event.shot_y_in) if event.shot_y_in == event.shot_y_in else "",
    }


def _write_rows(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _print_event_summary(tip_log, shot_log):
    print("\n  HIVE tips:")
    for alliance in ("red", "blue"):
        rows = [r for r in tip_log if r["alliance"] == alliance]
        in_match = [r for r in rows if r["phase"] in ("auto", "transition", "teleop")]
        auto = [r for r in rows if r["phase"] == "auto"]
        print("    {:<5} {} during match ({} in auto){}".format(
            alliance, len(in_match), len(auto),
            "" if len(rows) == len(in_match) else ", {} outside match time".format(len(rows) - len(in_match))))
    print("  Shots (made / missed):")
    for robot in ("R0", "R1", "R2", "R3", "unknown"):
        rows = [r for r in shot_log if r["robot"] == robot]
        made = sum(1 for r in rows if r["result"] == "made")
        print("    {}    {} / {}".format(robot, made, len(rows) - made))
    for hive in ("red", "blue"):
        made = sum(1 for r in shot_log if r["result"] == "made" and r["hive"] == hive)
        print("    into {} HIVE: {} made".format(hive, made))


def auto_calibrate_video(cv2, np, video_path, output_dir, start_s=0.0, end_s=None):
    """Run the automatic field calibration and save field_corners_auto.json
    plus a preview image in output_dir. Returns the calibration or None."""
    calibrator = FieldCalibrator(cv2, np)
    result = calibrator.calibrate_video(video_path, start_s=start_s, end_s=end_s)
    if result is None:
        print("[WARN] Automatic field calibration failed; draw corners with tools/calibrate.py.")
        return None
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "field_corners_auto.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = int(((start_s or 0.0) * fps + (end_s * fps if end_s else total - 1)) / 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, mid))
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(os.path.join(output_dir, "calibration_frame.jpg"), frame)
        cv2.imwrite(os.path.join(output_dir, "field_corners_auto_preview.jpg"),
                    draw_calibration(cv2, np, frame, result))
    a = result["auto"]
    print("[INFO] Field calibration: {} | corners visible {}/4 | confidence {}".format(
        a["method"], a["visible_corners"], a["confidence"]))
    for note in a.get("notes", []):
        print("[INFO]   {}".format(note))
    return result


def _print_instructions():
    print("""
+------------------------------------------------------------------+
|         AdvantageScope Visualization Instructions                |
+------------------------------------------------------------------+
|  1. Open AdvantageScope                                          |
|  2. File > Open Log(s) ... > select match_log.wpilog             |
|  3. "+" tab -> 2D Field -> set Field to FTC BIOBUZZ season       |
|  4. Drag Robot0/Pose into Poses                                  |
|  5. Click icon LEFT of the field name ->                         |
|       Format: Pose2d   Units: Meters + Radians                   |
|  6. Repeat for Robot1, Robot2, Robot3                            |
|  7. Press play!                                                  |
|                                                                  |
|  Robot IDs: assigned left-to-right at match start.               |
|  Debug overlay: 2-robot merges show CROSSED/ok; 3+ show          |
|  [entry_order->current_order] permutation string.                |
+------------------------------------------------------------------+
""")


def download_video(url, output_dir):
    import subprocess

    out = os.path.join(output_dir, "match_video.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print("[INFO] Reusing previously downloaded match video:", out)
        _emit_dashboard_stage("Loading Match Footage", "completed", "Reusing previously downloaded match footage")
        return out
    print("[INFO] Downloading:", url)
    _emit_dashboard_stage("Loading Match Footage", "in_progress", "Downloading match footage from YouTube")
    yt_dlp_cmd = find_yt_dlp_command()
    if not yt_dlp_cmd:
        _emit_dashboard_stage("Loading Match Footage", "failed", "yt-dlp is not available")
        print("[ERROR] yt-dlp is not installed for this Python and was not found on PATH.")
        sys.exit(1)
    yt_dlp_warning = get_yt_dlp_command_warning(yt_dlp_cmd)
    if yt_dlp_warning:
        print(yt_dlp_warning)
    attempts, attempt_notes = build_download_attempts(
        [Path.cwd(), Path(__file__).resolve().parent, Path(output_dir).resolve()]
    )
    for note in attempt_notes:
        print(note)
    last_recent_output = []

    for attempt_index, attempt in enumerate(attempts, start=1):
        if attempt_index > 1:
            print("[INFO] Retrying yt-dlp with {}...".format(attempt["label"]))
        bar = _make_bar("Downloading", 100)
        last_pct = 0
        recent_output = []
        proc = subprocess.Popen(
            [
                *yt_dlp_cmd,
                "--no-playlist",
                "--merge-output-format", "mp4",
                *yt_dlp_network_backoff_args(),
                "-o", out,
                *attempt["args"],
                url,
                "--progress-template", "%(progress._percent_str)s",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True)
        if proc.stdout:
            for line in proc.stdout:
                stripped = line.strip()
                if stripped:
                    recent_output.append(stripped)
                    recent_output = recent_output[-12:]
                try:
                    pct = int(float(stripped.rstrip("%")))
                    while last_pct < pct:
                        bar.next()
                        last_pct += 1
                    _emit_dashboard_stage_progress(
                        "Loading Match Footage",
                        "Downloading match footage from YouTube",
                        last_pct,
                        100,
                    )
                except ValueError:
                    pass
        proc.wait()
        while last_pct < 100:
            bar.next()
            last_pct += 1
            _emit_dashboard_stage_progress(
                "Loading Match Footage",
                "Downloading match footage from YouTube",
                last_pct,
                100,
            )
        bar.finish()
        last_recent_output = recent_output
        if proc.returncode == 0 and os.path.exists(out):
            _emit_dashboard_stage("Loading Match Footage", "completed", "Downloaded match footage from YouTube")
            return out
        if output_indicates_rate_limit(recent_output):
            print("[WARN] yt-dlp hit a YouTube 429 / Too Many Requests response. Stopping further retries to avoid additional throttling.")
            break

    if last_recent_output:
        _emit_dashboard_stage("Loading Match Footage", "failed", "Unable to download match footage from YouTube")
        print("[ERROR] yt-dlp failed. Last output:")
        for item in last_recent_output:
            print("  {}".format(item))
        joined_output = "\n".join(item.lower() for item in last_recent_output)
        blocked_by_youtube = (
            "sign in to confirm you're not a bot" in joined_output
            or "sign in to confirm you’re not a bot" in joined_output
            or "precondition check failed" in joined_output
            or "po token" in joined_output
        )
        tls_failed = (
            "certificate_verify_failed" in joined_output
            or "unable to get local issuer certificate" in joined_output
        )
        storyboards_only = (
            "only images are available for download" in joined_output
            or "requested format is not available" in joined_output
        )
        if tls_failed:
            print("[ERROR] Python TLS certificate verification failed. {}".format(
                tls_help_message()
            ))
        elif blocked_by_youtube:
            print("[ERROR] YouTube is challenging this session/IP. The anonymous retry paths were exhausted. {}".format(
                auth_help_message()
            ))
        elif storyboards_only:
            print("[ERROR] YouTube only exposed storyboard/blocked formats to the anonymous clients we tried. {}".format(
                auth_help_message()
            ))
        elif "nsig extraction failed" in joined_output:
            print("[ERROR] Your installed yt-dlp may be outdated. Updating yt-dlp often fixes this YouTube extractor error.")
        if output_indicates_rate_limit(last_recent_output):
            print("[ERROR] {}".format(rate_limit_help_message()))
        if "operation not permitted" in joined_output and "cookies" in joined_output:
            print("[ERROR] yt-dlp could not read a protected browser cookie store. {}".format(auth_help_message()))
    else:
        _emit_dashboard_stage("Loading Match Footage", "failed", "Unable to download match footage from YouTube")
        print("[ERROR] yt-dlp failed")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Track FTC {} robots from a match video.".format(season.GAME_NAME))
    parser.add_argument("url", nargs="?", help="YouTube URL")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--start-offset", type=float, default=0.0,
                        help="Seconds to skip before match timer starts")
    parser.add_argument("--auto-match-bounds", action="store_true",
                        help="Auto-detect match start/end from the broadcast timer")
    parser.add_argument("--sample-rate", type=float, default=10.0,
                        help="Frames per second to process (default 10)")
    parser.add_argument("--every-frame", dest="every_frame", action="store_true", default=None,
                        help="Track robots on every source frame (ignores --sample-rate; slower)")
    parser.add_argument("--no-every-frame", dest="every_frame", action="store_false",
                        help="Track robots at --sample-rate (balls and merges are still followed every frame)")
    parser.add_argument("--auto-only", action="store_true",
                        help="Track only AUTO (stops 2 s after it ends) for fast AUTO path scouting")
    parser.add_argument("--no-roi", action="store_true",
                        help="Process the whole frame instead of a window around the field")
    parser.add_argument("--debug", action="store_true",
                        help="Save annotated debug frames to tracker_debug/")
    parser.add_argument("--debug-video", action="store_true",
                        help="Write annotated debug output to tracker_debug.mp4 instead of tracker_debug/ images; implies --debug")
    parser.add_argument("--debug-every", type=int, default=1,
                        help="Save 1 debug frame every N processed frames (default 1)")
    parser.add_argument("--debug-enable-hitboxes", action="store_true",
                        help="Draw robot wireframe hitboxes in debug frames")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Only detect the field (corners, HIVE/FLOWER/overlay boxes), write "
                             "field_corners_auto.json + preview to --output-dir, and exit")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--corners", default=None,
                        help="field_corners.json from tools/calibrate.py (may also hold hive_boxes_px / exclude_px)")
    parser.add_argument("--robot-init-positions", default=None,
                        help=(
                            "JSON file or inline JSON string with starting field positions "
                            "for all 4 robots, sorted by robot id. "
                            "Format: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] in inches "
                            "with (0,0) at field center. "
                            "Bypasses blob-based init (required when robots start under "
                            "corner structures or are otherwise hard to detect at t=0). "
                            "Example: --robot-init-positions "
                            "'[[-52.7,-70.4],[-12.6,53.0],[15.8,60.6],[57.5,-55.7]]'"))
    parser.add_argument("--manual-reference-csv", default=None,
                        help=(
                            "Manual robot_positions-style CSV or JLOG used to build a supervised "
                            "appearance re-ID model for this video. "
                            "Coordinates are interpreted with (0,0) at field center. "
                            "Useful for tuning tracker identity assignment against "
                            "hand-labeled data."
                        ))
    args = parser.parse_args()

    if args.no_download:
        if not args.video_path:
            parser.error("--no-download requires --video-path")
        video_path = args.video_path
    else:
        if not args.url:
            parser.error("YouTube URL required (or --no-download --video-path)")
        os.makedirs(args.output_dir, exist_ok=True)
        video_path = download_video(args.url, args.output_dir)

    print("[INFO] Output:", args.output_dir)
    if args.calibrate_only:
        cv2 = _require("cv2", "opencv-python")
        np = _require("numpy")
        _emit_dashboard_stage("Loading Match Footage", "completed", "Match footage is ready")
        _emit_dashboard_stage("Calibrating Trackers", "in_progress", "Detecting field corners")
        result = auto_calibrate_video(cv2, np, video_path, args.output_dir, start_s=args.start_offset)
        if result is None:
            _emit_dashboard_stage("Calibrating Trackers", "failed", "Automatic field calibration failed")
            sys.exit(2)
        _emit_dashboard_stage("Calibrating Trackers", "completed", "Field calibration ready for approval")
        return
    corners = None
    calibration_extras = None
    if args.corners:
        with open(args.corners) as file:
            calibration = json.load(file)
        corners = calibration["corners_px"]
        calibration_extras = {
            key: calibration[key]
            for key in ("hive_boxes_px", "exclude_px", "flower_boxes_px")
            if key in calibration
        }
        print("[INFO] Corners:", args.corners)

    robot_init = None
    if args.robot_init_positions:
        raw = args.robot_init_positions.strip()
        if os.path.isfile(raw):
            with open(raw) as file:
                robot_init = json.load(file)
        else:
            robot_init = json.loads(raw)
        print("[INFO] Robot init positions:", robot_init)

    if args.no_roi:
        season.PROCESS_ROI = False
    process_match(
        video_path=video_path,
        output_dir=args.output_dir,
        start_offset_sec=args.start_offset,
        sample_rate_fps=args.sample_rate,
        process_every_frame=args.every_frame,
        auto_only=args.auto_only,
        debug=(args.debug or args.debug_video),
        debug_video=args.debug_video,
        debug_every_n=args.debug_every,
        debug_enable_hitboxes=args.debug_enable_hitboxes,
        manual_corners_px=corners,
        robot_init_positions=robot_init,
        manual_reference_csv=args.manual_reference_csv,
        auto_match_bounds=args.auto_match_bounds,
        calibration_extras=calibration_extras,
    )


if __name__ == "__main__":
    main()
