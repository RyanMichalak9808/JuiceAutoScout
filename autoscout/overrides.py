"""
Every module-level setting, editable from the dashboard Settings tab.

The settings are found by reading the source files (UPPERCASE module-level
assignments with literal values); the comment block above each one becomes
its help text, so a new setting added to one of these files shows up in the
dashboard automatically.

Changed values are passed to tracking jobs in the AUTOSCOUT_OVERRIDES
environment variable ({"season.PROCESS_ROI": false, ...}). Each module calls
apply() right after defining its settings, so the overrides are in place
before anything else reads them (including copies made at import time).
"""
import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

ENV_VAR = "AUTOSCOUT_OVERRIDES"
ROOT = Path(__file__).resolve().parent.parent

# module name -> source file
SOURCES = {
    "season": ROOT / "autoscout" / "season.py",
    "auto_scout": ROOT / "auto_scout.py",
    "stream_split": ROOT / "autoscout" / "stream_split.py",
    "field_detect": ROOT / "autoscout" / "field_detect.py",
}

# Not offered here: already a regular Settings-tab control, or not a setting.
SKIP = {
    "season.PROCESS_ROI",                    # Settings: Only process the field area
    "auto_scout.PROCESS_EVERY_SOURCE_FRAME",  # Settings: Track robots on every video frame
    "stream_split.CLIP_PRE_S",               # Settings: Clip lead-in before AUTO
    "stream_split.CLIP_POST_S",              # Settings: Clip tail after the match
    "auto_scout.DASHBOARD_EVENT_PREFIX",     # dashboard <-> job protocol marker
    "auto_scout.TIP_LOG_COLUMNS",            # CSV output format
    "auto_scout.SHOT_LOG_COLUMNS",           # CSV output format
}


def _kind(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "json"   # lists, tuples, dicts, None


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def discover() -> List[Dict[str, object]]:
    """All settings: [{key, module, name, kind, default, help, group}]."""
    out = []
    for module, path in SOURCES.items():
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        lines = text.splitlines()
        group = module
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id.isupper()):
                continue
            name = node.targets[0].id
            key = "{}.{}".format(module, name)
            try:
                default = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue   # computed value, not a plain setting
            # comment block directly above (skipping other assignments of the
            # same group, so grouped settings share their explanation)
            i = node.lineno - 2
            while i >= 0 and lines[i].strip() and not lines[i].lstrip().startswith("#"):
                i -= 1
            comment = []
            while i >= 0 and lines[i].lstrip().startswith("#"):
                comment.insert(0, lines[i].lstrip()[1:].strip())
                i -= 1
            heading = next((c for c in comment if c.startswith("──")), None)
            if heading:
                group = heading.strip("─ ").strip() or group
                comment = [c for c in comment if not c.startswith("──")]
            # trailing comment on the same line
            src = lines[node.lineno - 1]
            if "#" in src.split("=", 1)[-1]:
                comment.append(src.split("#", 1)[1].strip())
            if key in SKIP:
                continue
            out.append({"key": key, "module": module, "name": name, "kind": _kind(default),
                        "default": _jsonable(default), "help": " ".join(comment), "group": group})
    return out


def _coerce(value, default):
    """Bring a JSON value back to the default's type (tuples stay tuples)."""
    if value is None:
        return None
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, tuple) and isinstance(value, list):
        sample = default[0] if default else None
        return tuple(_coerce(v, sample) for v in value)
    if isinstance(default, list) and isinstance(value, list):
        sample = default[0] if default else None
        return [_coerce(v, sample) if isinstance(sample, (tuple, list)) else v for v in value]
    if isinstance(default, dict) and isinstance(value, dict):
        return {k: _coerce(v, default.get(k)) if k in default else v for k, v in value.items()}
    return value


def validate(overrides: Dict[str, object]) -> Dict[str, object]:
    """Keep only known keys, typed like their defaults; raise on bad values."""
    known = {s["key"]: s for s in discover()}
    clean = {}
    for key, value in (overrides or {}).items():
        spec = known.get(key)
        if spec is None:
            continue
        try:
            if spec["kind"] in ("int", "float") and value in (None, ""):
                raise ValueError("a number is required")
            typed = _coerce(value, spec["default"])
        except (TypeError, ValueError) as exc:
            raise ValueError("{}: {}".format(key, exc))
        if _jsonable(typed) != spec["default"]:
            clean[key] = _jsonable(typed)
    return clean


def current() -> Dict[str, object]:
    try:
        return json.loads(os.environ.get(ENV_VAR) or "{}")
    except ValueError:
        return {}


def apply(module: str, namespace: dict, overrides: Optional[Dict[str, object]] = None) -> None:
    """Set the overridden settings of one module (call at the end of its
    settings section with globals())."""
    prefix = module + "."
    for key, value in (current() if overrides is None else overrides).items():
        if key.startswith(prefix) and key[len(prefix):] in namespace:
            name = key[len(prefix):]
            try:
                namespace[name] = _coerce(value, namespace[name])
            except (TypeError, ValueError):
                pass
