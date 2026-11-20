"""
Scouting data from processed matches (for the dashboard Analytics page).

Each processed match folder (done.json) holds the tracker's CSVs. This
module reads them, links each match to its FTC Events teams, maps the
tracker's robots (R0-R3) to team numbers and totals the stats per team.

Robot -> team mapping:
- confirmed: team_map.json in the match folder, saved from the dashboard;
- otherwise a guess: a robot's alliance is the half of the field it starts
  AUTO on (red left, blue right), or else the HIVE it shoots into most, and
  the alliance's teams are handed out in robot order. Guesses are
  marked so they can be checked in the match menu.
"""
import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from autoscout import auto_paths

TEAM_MAP_FILE = "team_map.json"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return []


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _num(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _match_label(match_dir: Path) -> str:
    for name in ("job_payload.json", "stream_clip.json"):
        info = _read_json(match_dir / name) or {}
        label = info.get("match_label") or info.get("label")
        if label:
            return str(label)
    return match_dir.name.replace("-", " ").title()


def _guess_team_map(robots: List[dict], red: List[str], blue: List[str]) -> Dict[str, Optional[str]]:
    """Robots grouped by guessed alliance, then given that alliance's teams in
    robot order; robots left over get the teams still free."""
    mapping: Dict[str, Optional[str]] = {r["robot"]: None for r in robots}
    free = {"red": list(red), "blue": list(blue)}
    for alliance in ("red", "blue"):
        for r in robots:
            if r["alliance_guess"] == alliance and free[alliance]:
                mapping[r["robot"]] = free[alliance].pop(0)
    rest = free["red"] + free["blue"]
    for r in robots:
        if mapping[r["robot"]] is None and rest:
            mapping[r["robot"]] = rest.pop(0)
    return mapping


def load_match(match_dir: Path, ftc_match: Optional[dict], url_for: Callable[[Path], Optional[str]]) -> Optional[dict]:
    match_dir = Path(match_dir)
    if not (match_dir / "done.json").exists() or not (match_dir / "robot_summary.csv").exists():
        return None
    red = [str(t) for t in (ftc_match or {}).get("red_teams") or []]
    blue = [str(t) for t in (ftc_match or {}).get("blue_teams") or []]
    shots = [{
        "t": _num(s.get("match_time_s")), "phase": s.get("phase"), "robot": s.get("robot"),
        "result": s.get("result"), "hive": s.get("hive"),
        "x": _num(s.get("shooter_x_in"), None), "y": _num(s.get("shooter_y_in"), None),
    } for s in _read_csv(match_dir / "shot_events.csv")]
    robots = []
    for row in _read_csv(match_dir / "robot_summary.csv"):
        rid = row.get("robot")
        hives = [s["hive"] for s in shots if s["robot"] == rid and s["hive"] in ("red", "blue")]
        guess = max(("red", "blue"), key=hives.count) if hives else None
        robots.append({
            "robot": rid,
            "alliance_guess": guess,
            "seen_s": _num(row.get("seen_s")),
            "distance_in": _num(row.get("distance_in")),
            "moving_pct": _num(row.get("moving_pct")),
            "longest_stall_s": _num(row.get("longest_stall_s")),
            "field_coverage_pct": _num(row.get("field_coverage_pct")),
            "shots_made": int(_num(row.get("shots_made"))),
            "shots_missed": int(_num(row.get("shots_missed"))),
            "auto_made": sum(1 for s in shots if s["robot"] == rid and s["result"] == "made" and s["phase"] == "auto"),
            "red_flags": row.get("red_flags") or "",
        })
    autos = {a["robot"]: a for a in (auto_paths.load_or_build(str(match_dir)) or [])}
    for r in robots:
        r["auto"] = autos.get(r["robot"])
        # Alliance guess: each alliance starts on its own half (red on the
        # left, x < 0, in the field frame), which beats the HIVE-based guess.
        start = (r["auto"] or {}).get("start")
        if start and (r["auto"] or {}).get("seen_from_s", 99) <= 3.0 and abs(start[0]) >= 6.0:
            r["alliance_guess"] = "red" if start[0] < 0 else "blue"
    saved = _read_json(match_dir / TEAM_MAP_FILE)
    team_map = {r["robot"]: (saved or {}).get(r["robot"]) for r in robots} if saved else _guess_team_map(robots, red, blue)
    for r in robots:
        r["team"] = team_map.get(r["robot"])
        r["alliance"] = "red" if r["team"] in red else "blue" if r["team"] in blue else r["alliance_guess"]
    files = {}
    for name in ("alliance_summary.csv", "robot_summary.csv", "shot_events.csv", "hive_tips.csv",
                 "robot_positions.csv", "resource_density.csv", "auto_paths.csv",
                 "shot_heatmap.png", "resource_density.png", "auto_paths.png"):
        if (match_dir / name).exists():
            files[name] = url_for(match_dir / name)
    return {
        "dir": match_dir.name,
        "label": _match_label(match_dir),
        "red_teams": red,
        "blue_teams": blue,
        "red_score": (ftc_match or {}).get("red_score"),
        "blue_score": (ftc_match or {}).get("blue_score"),
        "team_map_confirmed": bool(saved),
        "auto_only": bool((_read_json(match_dir / "done.json") or {}).get("auto_only")),
        "robots": robots,
        "alliances": _read_csv(match_dir / "alliance_summary.csv"),
        "tips": [{"alliance": t.get("alliance"), "t": _num(t.get("match_time_s")), "clock": t.get("match_clock"),
                  "phase": t.get("phase")} for t in _read_csv(match_dir / "hive_tips.csv")],
        "shots": shots,
        "files": files,
    }


def team_totals(matches: List[dict]) -> List[dict]:
    """Per-team scouting stats over the processed matches."""
    teams: Dict[str, dict] = {}
    for m in matches:
        if m.get("auto_only"):
            continue   # only AUTO was tracked: its shots/distance are not a whole match
        tips_by_alliance = {a.get("alliance"): _num(a.get("tips")) for a in m["alliances"]}
        for r in m["robots"]:
            team = r.get("team")
            if not team:
                continue
            t = teams.setdefault(team, {"team": team, "matches": 0, "confirmed": 0, "made": 0, "missed": 0,
                                        "auto_made": 0, "distance": 0.0, "moving": 0.0, "flags": 0,
                                        "alliance_tips": 0.0, "match_labels": []})
            t["matches"] += 1
            t["confirmed"] += 1 if m["team_map_confirmed"] else 0
            t["made"] += r["shots_made"]
            t["missed"] += r["shots_missed"]
            t["auto_made"] += r["auto_made"]
            t["distance"] += r["distance_in"]
            t["moving"] += r["moving_pct"]
            t["flags"] += 1 if r["red_flags"] else 0
            t["alliance_tips"] += tips_by_alliance.get(r["alliance"], 0.0)
            t["match_labels"].append(m["label"])
    out = []
    for t in teams.values():
        n = max(t["matches"], 1)
        shots = t["made"] + t["missed"]
        out.append({
            "team": t["team"],
            "matches": t["matches"],
            "confirmed": t["confirmed"],
            "made_per_match": round(t["made"] / n, 1),
            "missed_per_match": round(t["missed"] / n, 1),
            "accuracy_pct": round(100.0 * t["made"] / shots) if shots else None,
            "auto_made_per_match": round(t["auto_made"] / n, 1),
            "distance_ft_per_match": round(t["distance"] / n / 12.0),
            "moving_pct": round(t["moving"] / n),
            "red_flag_matches": t["flags"],
            "alliance_tips_per_match": round(t["alliance_tips"] / n, 1),
            "match_labels": t["match_labels"],
        })
    out.sort(key=lambda r: r["made_per_match"], reverse=True)
    return out


def team_autos(matches: List[dict]) -> Dict[str, dict]:
    """Per team: its AUTO path in every processed match (full or AUTO-only),
    plus how consistent the paths are (mean distance between paths, inches)."""
    out: Dict[str, dict] = {}
    for m in matches:
        for r in m["robots"]:
            team, auto = r.get("team"), r.get("auto")
            if not team or not auto or not auto.get("points"):
                continue
            t = out.setdefault(team, {"team": team, "runs": []})
            t["runs"].append({"match": m["label"], "dir": m["dir"], "alliance": r.get("alliance"),
                              "robot": r["robot"], "confirmed": m["team_map_confirmed"],
                              "start": auto.get("start"), "end": auto.get("end"),
                              "distance_in": auto.get("distance_in"), "moved": auto.get("moved"),
                              "visible_pct": auto.get("visible_pct"), "points": auto["points"]})
    for t in out.values():
        runs = t["runs"]
        sims = [auto_paths.path_similarity(a["points"], b["points"])
                for i, a in enumerate(runs) for b in runs[i + 1:]]
        sims = [s for s in sims if s is not None]
        starts = [r["start"] for r in runs if r["start"]]
        cx = sum(s[0] for s in starts) / len(starts) if starts else None
        cy = sum(s[1] for s in starts) / len(starts) if starts else None
        t["matches"] = len(runs)
        t["moved_pct"] = round(100.0 * sum(1 for r in runs if r["moved"]) / len(runs)) if runs else None
        t["avg_distance_in"] = round(sum(r["distance_in"] or 0 for r in runs) / len(runs)) if runs else None
        t["path_spread_in"] = round(sum(sims) / len(sims)) if sims else None
        t["start_spread_in"] = (round(sum(((s[0] - cx) ** 2 + (s[1] - cy) ** 2) ** 0.5 for s in starts) / len(starts))
                                if len(starts) > 1 else None)
    return out


def load_event(event_root: Path, ftc_matches: List[dict], url_for: Callable[[Path], Optional[str]]) -> dict:
    by_label = {str(m.get("label")): m for m in ftc_matches or []}
    matches = []
    if Path(event_root).exists():
        for d in sorted(p for p in Path(event_root).iterdir() if p.is_dir()):
            label = _match_label(d)
            m = load_match(d, by_label.get(label), url_for)
            if m:
                matches.append(m)

    def order(m):
        digits = "".join(ch for ch in m["label"] if ch.isdigit())
        return (0 if m["label"].lower().startswith("qual") else 1, int(digits or 0), m["label"])
    matches.sort(key=order)
    return {"matches": matches, "teams": team_totals(matches), "team_autos": team_autos(matches)}


def save_team_map(match_dir: Path, mapping: Dict[str, Optional[str]]) -> None:
    clean = {str(k): (str(v).strip() or None) if v not in (None, "") else None for k, v in mapping.items()}
    tmp = Path(match_dir) / (TEAM_MAP_FILE + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2)
    tmp.replace(Path(match_dir) / TEAM_MAP_FILE)
