"""Shared modules for the AutoScout pipeline."""
import sys as _sys

# A Windows console (cp1252) cannot print some characters; print '?' for
# them instead of stopping the run with UnicodeEncodeError.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
