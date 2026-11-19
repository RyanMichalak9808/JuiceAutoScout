"""
Split a full event stream into per-match clips.

1. Audio only (cheap): find the field's end-of-AUTO and end-of-match buzzers.
   Both are a ~3240 Hz tone, and they always come exactly 128 s apart
   (8 s transition + 2:00 TELEOP). A narrow-band tone test ignores
   commentary, and the 128 s pairing ignores stray beeps.
   AUTO starts AUTO_START_BEFORE_BUZZER_S before the end-of-AUTO buzzer.
2. Label: line the detected matches up with the FTC Events start times.
   Those times are only to the minute and often off, so only the gaps between
   matches are used (best single offset + monotone alignment with skips).
3. Cut a padded clip per match (ffmpeg stream copy or a yt-dlp section
   download, no re-encoding) and find the buzzer again in the clip's own
   audio, so the AUTO start offset in the clip is exact even if the cut
   landed on an earlier keyframe.
"""
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

SR = 8000
HOP = 400                       # 0.05 s analysis step
NFFT = 1024
TONE_HZ = (3200.0, 3290.0)      # field buzzer
REF_HZ = ((2800.0, 3100.0), (3400.0, 3700.0))
TONE_RATIO = 20.0               # tone bin vs neighbouring bands
MIN_TONE_S = 1.5                # buzzers last 2-3 s; beeps and speech are shorter
PAIR_GAP_S = 128.0              # end-of-AUTO -> end-of-match
PAIR_TOL_S = 1.5
AUTO_START_BEFORE_BUZZER_S = 29.6
MATCH_LEN_S = 158.0             # 30 AUTO + 8 transition + 120 TELEOP
CLIP_PAD_S = 6.0
LABEL_TOL_S = 180.0             # schedule times are minute-rounded and drift
LABEL_SKIP_COST_S = 240.0


def _media_duration(path: str) -> float:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
        return float(out.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def load_audio(path: str, workers: Optional[int] = None) -> np.ndarray:
    """Mono 8 kHz int16 audio of any file ffmpeg can read. Long files are
    decoded in parallel time chunks (decoding is the slow part)."""
    def run(args):
        return subprocess.run(["ffmpeg", "-v", "error", *args, "-i", str(path), "-vn", "-ac", "1",
                               "-ar", str(SR), "-f", "s16le", "-"], capture_output=True, check=True).stdout
    duration = _media_duration(path)
    workers = workers or max(1, min(8, os.cpu_count() or 1))
    if duration < 600 or workers == 1:
        return np.frombuffer(run([]), dtype=np.int16)
    chunk = duration / workers
    with ThreadPoolExecutor(workers) as pool:
        parts = list(pool.map(lambda k: run(["-ss", "{:.3f}".format(k * chunk), "-t", "{:.3f}".format(chunk)]),
                              range(workers)))
    return np.frombuffer(b"".join(parts), dtype=np.int16)


def _tone_ratio(x: np.ndarray) -> np.ndarray:
    if len(x) < NFFT:
        return np.zeros(0, np.float32)
    win = np.hanning(NFFT).astype(np.float32)
    view = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP]
    b = lambda f: int(round(f * NFFT / SR))
    out = np.empty(len(view), np.float32)
    for i in range(0, len(view), 20000):
        P = np.abs(np.fft.rfft(view[i:i + 20000].astype(np.float32) * win, axis=1))
        tone = P[:, b(TONE_HZ[0]):b(TONE_HZ[1]) + 1].max(1)
        ref = np.median(np.hstack([P[:, b(lo):b(hi)] for lo, hi in REF_HZ]), 1) + 1e-6
        out[i:i + 20000] = tone / ref
    return out


def find_buzzers(x: np.ndarray) -> List[Dict[str, float]]:
    """Long narrow-band tone events: [{t, dur, strength}] (t = onset, s)."""
    r = _tone_ratio(x)
    on = r > TONE_RATIO
    step = HOP / SR
    events, i, n = [], 0, len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j:j + 10].any():   # bridge gaps < 0.5 s
            j += 1
        if (j - i) * step >= MIN_TONE_S:
            events.append({"t": i * step, "dur": (j - i) * step, "strength": float(r[i:j].max())})
        i = j
    return events


def detect_matches(x: np.ndarray) -> List[Dict[str, object]]:
    """Matches in the audio, in stream order. confidence: 'high' when both
    buzzers were heard 128 s apart, 'low' when only one was."""
    events = find_buzzers(x)
    used = set()
    found = []
    for a, ea in enumerate(events):
        if a in used:
            continue
        partner = next((b for b in range(a + 1, len(events)) if b not in used
                        and abs(events[b]["t"] - ea["t"] - PAIR_GAP_S) <= PAIR_TOL_S), None)
        if partner is not None:
            used.update((a, partner))
            found.append((ea["t"], "high"))
    # lone buzzers (the other one drowned out): keep the strong ones
    for a, ea in enumerate(events):
        if a in used or ea["strength"] < 3 * TONE_RATIO:
            continue
        auto_end = ea["t"] if ea["dur"] < 2.6 else ea["t"] - PAIR_GAP_S
        if all(abs(auto_end - t) > MATCH_LEN_S for t, _c in found) and auto_end > 0:
            found.append((auto_end, "low"))
    found.sort()
    return [{"auto_end_s": round(t, 2),
             "start_s": round(t - AUTO_START_BEFORE_BUZZER_S, 2),
             "end_s": round(t - AUTO_START_BEFORE_BUZZER_S + MATCH_LEN_S, 2),
             "confidence": c} for t, c in found]


_TIME_RE = re.compile(r"(\d{1,2})/(\d{1,2})\s*-\s*(\d{1,2}):(\d{2})\s*([AP]M)", re.I)


def parse_schedule_time(text: Optional[str]) -> Optional[float]:
    """'Sat 11/14 - 2:44 PM' -> seconds (only differences are used)."""
    m = _TIME_RE.search(text or "")
    if not m:
        return None
    mo, d, h, mi, ap = int(m[1]), int(m[2]), int(m[3]) % 12, int(m[4]), m[5].upper()
    h += 12 if ap == "PM" else 0
    return (datetime(2000, mo, d, h, mi) - datetime(2000, 1, 1)).total_seconds()


def _align(det: Sequence[float], sch: Sequence[float], offset: float):
    """Monotone alignment det[i] <-> sch[j] + offset with skips. (cost, pairs)"""
    n, m = len(det), len(sch)
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == INF:
                continue
            if i < n and c + LABEL_SKIP_COST_S < cost[i + 1][j]:
                cost[i + 1][j], back[i + 1][j] = c + LABEL_SKIP_COST_S, (i, j, False)
            if j < m and c + LABEL_SKIP_COST_S < cost[i][j + 1]:
                cost[i][j + 1], back[i][j + 1] = c + LABEL_SKIP_COST_S, (i, j, False)
            if i < n and j < m:
                d = abs(det[i] - sch[j] - offset)
                if d <= LABEL_TOL_S and c + d < cost[i + 1][j + 1]:
                    cost[i + 1][j + 1], back[i + 1][j + 1] = c + d, (i, j, True)
    pairs, i, j = [], n, m
    while back[i][j] is not None:
        pi, pj, matched = back[i][j]
        if matched:
            pairs.append((pi, pj))
        i, j = pi, pj
    return cost[n][m], pairs[::-1]


def label_matches(detected: List[Dict[str, object]], schedule: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """schedule: [{label, phase, time}] from FTC Events (time = raw text).
    Adds 'label' / 'phase' / 'schedule_error_s' to each detected match."""
    sched = [(parse_schedule_time(s.get("time")), s) for s in schedule]
    sched = sorted([(t, s) for t, s in sched if t is not None], key=lambda p: p[0])
    det = [float(d["start_s"]) for d in detected]
    for d in detected:
        d.update(label=None, phase=None, schedule_error_s=None)
    if not det:
        return detected
    if not sched:
        # no times: fall back to order
        for d, s in zip(detected, schedule):
            d.update(label=s.get("label"), phase=s.get("phase"))
        return detected
    times = [t for t, _s in sched]
    best = None
    for i in range(len(det)):
        for j in range(len(times)):
            off = det[i] - times[j]
            cost, pairs = _align(det, times, off)
            if pairs:
                off = float(np.median([det[a] - times[b] for a, b in pairs]))
                cost, pairs = _align(det, times, off)
            if best is None or cost < best[0]:
                best = (cost, pairs, off)
    _cost, pairs, off = best
    for a, b in pairs:
        s = sched[b][1]
        detected[a].update(label=s.get("label"), phase=s.get("phase"),
                           schedule_error_s=round(det[a] - times[b] - off, 1))
    return detected


def relocate_start(clip_path: str, expected_start_s: float) -> Optional[float]:
    """AUTO start inside a cut clip, from the clip's own end-of-AUTO buzzer."""
    try:
        x = load_audio(clip_path)
    except subprocess.CalledProcessError:
        return None
    want = expected_start_s + AUTO_START_BEFORE_BUZZER_S
    near = [e for e in find_buzzers(x) if abs(e["t"] - want) <= 10.0]
    if not near:
        return None
    best = min(near, key=lambda e: abs(e["t"] - want))
    return round(best["t"] - AUTO_START_BEFORE_BUZZER_S, 2)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")


def split_stream(source: str, schedule: List[Dict[str, object]], out_root: Path,
                 yt_dlp_cmd: Optional[List[str]] = None, yt_dlp_attempts: Optional[List[List[str]]] = None,
                 log: Callable[[str], None] = print, cut: bool = True) -> Dict[str, object]:
    """source: local video/audio file or a YouTube URL. Writes
    out_root/<slug(label)>/match_video.mp4 + stream_clip.json and
    out_root/stream_split.json. Returns the summary."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    is_url = bool(re.match(r"https?://", source))
    attempts = yt_dlp_attempts or [[]]

    def ytdlp(extra: List[str]) -> None:
        last = ""
        for args in attempts:
            proc = subprocess.run([*yt_dlp_cmd, *args, *extra, source], capture_output=True, text=True)
            if proc.returncode == 0:
                return
            last = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
            last = last[0]
        raise RuntimeError("yt-dlp failed: {}".format(last))

    if is_url:
        audio_path = out_root / "stream_audio.webm"
        if not audio_path.exists():
            log("[INFO] Downloading stream audio (audio only)...")
            ytdlp(["-f", "wa/ba", "-o", str(audio_path)])
    else:
        audio_path = Path(source)
    log("[INFO] Finding match buzzers in the audio...")
    detected = detect_matches(load_audio(str(audio_path)))
    log("[INFO] Found {} matches in the stream.".format(len(detected)))
    label_matches(detected, schedule)

    for d in detected:
        if not d.get("label"):
            d["status"] = "unlabeled"
            continue
        clip_dir = out_root / slugify(d["label"])
        clip = clip_dir / "match_video.mp4"
        a = max(0.0, float(d["start_s"]) - CLIP_PAD_S)
        b = float(d["end_s"]) + CLIP_PAD_S
        d["clip_path"] = str(clip)
        if not cut:
            d["status"] = "detected"
            continue
        clip_dir.mkdir(parents=True, exist_ok=True)
        if not clip.exists():
            log("[INFO] Cutting {} ({:.0f}s-{:.0f}s)...".format(d["label"], a, b))
            tmp = clip_dir / "match_video.part.mp4"
            if is_url:
                ytdlp(["-f", "bv*[height<=720]+ba/b[height<=720]/b", "--merge-output-format", "mp4",
                       "--download-sections", "*{:.1f}-{:.1f}".format(a, b), "-o", str(tmp)])
            else:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "{:.2f}".format(a), "-i", source,
                                "-t", "{:.2f}".format(b - a), "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                                "-avoid_negative_ts", "make_zero", str(tmp)], check=True)
            tmp.replace(clip)
        expected = float(d["start_s"]) - a
        exact = relocate_start(str(clip), expected)
        d["start_offset"] = exact if exact is not None else round(expected, 2)
        d["start_from"] = "clip audio" if exact is not None else "stream audio"
        d["status"] = "clip_ready"
        with (clip_dir / "stream_clip.json").open("w", encoding="utf-8") as fh:
            json.dump({"source": source, **d}, fh, indent=2)

    summary = {"source": source, "matches": detected,
               "labeled": sum(1 for d in detected if d.get("label")),
               "low_confidence": sum(1 for d in detected if d["confidence"] != "high")}
    with (out_root / "stream_split.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary
