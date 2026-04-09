"""
Batch-prepare the current speaker library in place.

This script backs up the active speaker files, applies the current speaker-prep
heuristics to each `.wav`, and replaces the active file with the cleaned result
under the same filename so the app uses the better versions going forward.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from backend.api.core.app_paths import SPEAKERS_DIR
from backend.api.core.source_clip_prep import analyze_source_clip, prepare_source_clip


REPORTS_DIR = Path("shared/data/speaker_prep_reports")
BACKUPS_ROOT = Path("shared/audio/speakers_backups")
TEMP_ROOT = Path("shared/audio/temp/speaker_prep_batch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-prepare the current speaker library in place.")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report planned prep without changing files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick trial runs.")
    return parser


def to_pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def list_speaker_files(limit: int | None = None) -> List[Path]:
    files = sorted(SPEAKERS_DIR.glob("*.wav"))
    if limit is not None:
        return files[: max(limit, 0)]
    return files


def build_report_entry(filename: str, before: Dict[str, Any], result: Dict[str, Any] | None = None) -> Dict[str, Any]:
    entry = {
        "filename": filename,
        "before": {
            "score": before.get("clone_readiness_score"),
            "label": before.get("clone_readiness_label"),
            "duration_seconds": before.get("duration_seconds"),
            "channels": before.get("channels"),
            "level_dbfs": before.get("level_dbfs"),
            "peak_dbfs": before.get("peak_dbfs"),
            "silence_percent": before.get("silence_percent"),
            "suggested_prep": before.get("suggested_prep"),
        },
    }
    if result is not None:
        after = result.get("after", {})
        entry["after"] = {
            "score": after.get("clone_readiness_score"),
            "label": after.get("clone_readiness_label"),
            "duration_seconds": after.get("duration_seconds"),
            "channels": after.get("channels"),
            "level_dbfs": after.get("level_dbfs"),
            "peak_dbfs": after.get("peak_dbfs"),
            "silence_percent": after.get("silence_percent"),
        }
        entry["processing_notes"] = result.get("processing_notes", [])
    return entry


def main() -> int:
    args = build_parser().parse_args()
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    speaker_files = list_speaker_files(args.limit)

    if not speaker_files:
        print("No .wav speaker files found.")
        return 0

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    backup_dir = BACKUPS_ROOT / timestamp
    temp_dir = TEMP_ROOT / timestamp
    report_path = REPORTS_DIR / f"speaker-prep-batch-{timestamp}.json"

    report: Dict[str, Any] = {
        "created_at_utc": timestamp,
        "dry_run": bool(args.dry_run),
        "speaker_count": len(speaker_files),
        "backup_dir": str(backup_dir),
        "entries": [],
    }

    if not args.dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

    for speaker_path in speaker_files:
        before = analyze_source_clip(speaker_path)
        suggested = before.get("suggested_prep", {})
        entry = build_report_entry(speaker_path.name, before)

        if args.dry_run:
            report["entries"].append(entry)
            continue

        backup_target = backup_dir / speaker_path.name
        temp_output = temp_dir / speaker_path.name
        shutil.copy2(speaker_path, backup_target)

        result = prepare_source_clip(
            speaker_path,
            temp_output,
            start_time=suggested.get("start_time"),
            end_time=suggested.get("end_time"),
            convert_to_mono=bool(suggested.get("convert_to_mono", True)),
            normalize_audio=bool(suggested.get("normalize_audio", True)),
            target_peak_dbfs=float(suggested.get("target_peak_dbfs", -1.0)),
            use_noise_reduction=bool(suggested.get("use_noise_reduction", False)),
            use_vocal_separation=bool(suggested.get("use_vocal_separation", False)),
        )

        shutil.move(str(temp_output), str(speaker_path))
        entry = build_report_entry(speaker_path.name, before, result)
        report["entries"].append(entry)

    report_path.write_text(to_pretty_json(report), encoding="utf-8")

    summary = {
        "processed": len(report["entries"]),
        "dry_run": bool(args.dry_run),
        "report": str(report_path),
    }
    if not args.dry_run:
        summary["backup_dir"] = str(backup_dir)

    print(to_pretty_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
