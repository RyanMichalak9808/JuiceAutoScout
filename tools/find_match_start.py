"""
Suggest the match start time for --start-offset without trusting the
broadcast clock.

    python3 tools/find_match_start.py match.mp4 --corners field_corners.json

Prints candidate start times (first parked robot driving away, first POLLEN
moving inside a HIVE box) and saves a strip of frames around each candidate
(match_start_check.png) so you can confirm by eye: autonomous starts at the
first frame where a robot drives or shoots.
"""
import argparse
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autoscout.match_start import MatchStartDetector  # noqa: E402
from autoscout.tracker import RobotTracker  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--corners", required=True)
    parser.add_argument("--max-scan", type=float, default=45.0, help="seconds of video to scan")
    parser.add_argument("--output", default="match_start_check.png")
    args = parser.parse_args()

    with open(args.corners) as handle:
        calibration = json.load(handle)
    bl, br, tr, tl = calibration["corners_px"]
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Cannot read {}".format(args.video))
    tracker = RobotTracker(cv2, np)
    with contextlib.redirect_stdout(io.StringIO()):
        tracker.setup(args.video, np.array([tl, tr, br, bl], dtype=np.float32), frame.shape, calibration)
    result = MatchStartDetector(cv2, np).detect(args.video, tracker, args.max_scan)
    candidates = result.get("candidates") or []
    if not candidates:
        print("No start candidate found in the first {:.0f} s.".format(args.max_scan))
        return
    for reason, t in sorted(candidates, key=lambda c: c[1]):
        print("  {:6.1f} s  {}".format(t, reason))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    for reason, t in sorted(candidates, key=lambda c: c[1]):
        tiles = []
        for dt in (-3.0, -2.0, -1.0, 0.0, 1.0):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(max(0.0, t + dt) * fps)))
            ok, f = cap.read()
            if not ok:
                f = np.zeros_like(frame)
            f = cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1])))
            cv2.putText(f, "{:.1f}s".format(t + dt), (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            tiles.append(f)
        rows.append(np.hstack(tiles))
    cv2.imwrite(args.output, np.vstack(rows))
    print("Frames around each candidate: {}".format(args.output))
    print("Use the time where robots first move or shoot as --start-offset.")


if __name__ == "__main__":
    main()
