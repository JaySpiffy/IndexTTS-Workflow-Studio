"""Compatibility package that exposes the bundled IndexTTS modules."""

from pathlib import Path

_package_dir = Path(__file__).resolve().parent
_backend_indextts_dir = _package_dir.parent / "backend" / "indextts"

# Search both the compatibility package and the bundled backend modules.
__path__ = [str(_package_dir)]
if _backend_indextts_dir.exists():
    __path__.append(str(_backend_indextts_dir))
