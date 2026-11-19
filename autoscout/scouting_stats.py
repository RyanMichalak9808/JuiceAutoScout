"""
Team statistics from FTC Events match results (port of the team's scouting
calculator, scoutingcalculatorfixed 5-3-26, without DPR and CCWM).

Per team:
- OPR     least-squares offensive power rating (alliance score = sum of OPRs)
- EPA     ridge-regularized (lambda 10) rating from score margins
- RSI     OPR as a z-score across the event
- oprSOS  mean OPR of the opponents faced (bigger = harder schedule)
- epaSOS  mean EPA of the opponents faced (lower = harder, as in the script)
- Record  wins-losses-ties, Win%

Only matches with both scores and four team numbers are used (by default the
qualification matches, like the calculator's Q- input).
"""
import re
from typing import Dict, Iterable, List

import numpy as np

EPA_LAMBDA = 10.0


def _team(value) -> int:
    return int(re.sub(r"\D", "", str(value)))


def compute_team_stats(matches: Iterable[Dict[str, object]], phases=("qualifications",)) -> Dict[str, object]:
    played = []
    for m in matches:
        if phases and m.get("phase") not in phases:
            continue
        red, blue = m.get("red_teams") or [], m.get("blue_teams") or []
        rs, bs = m.get("red_score"), m.get("blue_score")
        if len(red) < 2 or len(blue) < 2 or rs is None or bs is None:
            continue
        try:
            played.append(([_team(t) for t in red[:2]], [_team(t) for t in blue[:2]], int(rs), int(bs),
                           str(m.get("label") or "")))
        except ValueError:
            continue
    if not played:
        return {"teams": [], "matches_used": 0, "note": "No played matches with scores yet."}

    teams = sorted({t for r, b, _rs, _bs, _l in played for t in r + b})
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    A_off, b_off, A_margin, b_margin = [], [], [], []
    wins = {t: 0 for t in teams}
    losses = {t: 0 for t in teams}
    ties = {t: 0 for t in teams}
    for red, blue, rs, bs, _label in played:
        for own, opp, own_s, opp_s in ((red, blue, rs, bs), (blue, red, bs, rs)):
            row = np.zeros(n)
            for t in own:
                row[index[t]] = 1
            A_off.append(row)
            b_off.append(own_s)
            margin = np.zeros(n)
            for t in own:
                margin[index[t]] += 1
            for t in opp:
                margin[index[t]] -= 1
            A_margin.append(margin)
            b_margin.append(own_s - opp_s)
        if rs > bs:
            for t in red:
                wins[t] += 1
            for t in blue:
                losses[t] += 1
        elif bs > rs:
            for t in blue:
                wins[t] += 1
            for t in red:
                losses[t] += 1
        else:
            for t in red + blue:
                ties[t] += 1

    A_off, b_off = np.array(A_off), np.array(b_off, dtype=float)
    A_margin, b_margin = np.array(A_margin), np.array(b_margin, dtype=float)
    opr = np.linalg.lstsq(A_off, b_off, rcond=None)[0]
    epa = np.linalg.solve(A_margin.T @ A_margin + EPA_LAMBDA * np.eye(n), A_margin.T @ b_margin)
    std = float(np.std(opr))
    rsi = (opr - np.mean(opr)) / std if std != 0 else np.zeros_like(opr)

    rows: List[Dict[str, object]] = []
    for t in teams:
        opp_opr, opp_epa = [], []
        for red, blue, _rs, _bs, _l in played:
            opponents = blue if t in red else red if t in blue else None
            if opponents is None:
                continue
            for o in opponents:
                opp_opr.append(opr[index[o]])
                opp_epa.append(epa[index[o]])
        games = wins[t] + losses[t] + ties[t]
        i = index[t]
        rows.append({
            "team": t,
            "EPA": round(float(epa[i]), 2),
            "OPR": round(float(opr[i]), 2),
            "RSI": round(float(rsi[i]), 2),
            "oprSOS": round(float(np.mean(opp_opr)), 2) if opp_opr else 0.0,
            "epaSOS": round(float(np.mean(opp_epa)), 2) if opp_epa else 0.0,
            "record": "{}-{}-{}".format(wins[t], losses[t], ties[t]),
            "win_pct": round(wins[t] / games, 2) if games else 0.0,
            "matches": games,
        })
    rows.sort(key=lambda r: r["EPA"], reverse=True)
    return {
        "teams": rows,
        "matches_used": len(played),
        "matches": [{"label": l, "red": r, "blue": b, "red_score": rs, "blue_score": bs}
                    for r, b, rs, bs, l in played],
        "note": "",
    }
