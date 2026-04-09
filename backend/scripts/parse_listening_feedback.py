"""
Parse compact human listening feedback into a structured JSON summary.

Example input:

    CLIP=b4a5cd17/L1/V2
    VERDICT=bad
    SIMILARITY=2
    NATURALNESS=2
    PACE=1
    ROBOTIC=5
    CLARITY=3
    EMOTION=2
    ISSUES=too_fast,robotic,weak_similarity
    ACTION=more_faithful,slower,cleaner_ref
    NOTES=Voice is rushed and doesn't sound enough like the reference.

Usage inside the backend container:

    python /app/backend/scripts/parse_listening_feedback.py /app/feedback.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCORE_FIELDS = ("SIMILARITY", "NATURALNESS", "PACE", "ROBOTIC", "CLARITY", "EMOTION")
LIST_FIELDS = ("ISSUES", "ACTION")
TEXT_FIELDS = ("CLIP", "VERDICT", "NOTES")
KEY_ALIASES = {
    "OVERALL": "VERDICT",
    "ACTIONS": "ACTION",
    "ISSUE": "ISSUES",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse human listening feedback into JSON")
    parser.add_argument("feedback_file", help="Path to a text file containing one or more feedback blocks")
    parser.add_argument("--output", help="Optional path to write the parsed JSON summary")
    return parser.parse_args()


def normalize_key(raw_key: str) -> str:
    key = raw_key.strip().upper()
    return KEY_ALIASES.get(key, key)


def parse_score(raw_value: str) -> int:
    cleaned = raw_value.strip()
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0].strip()

    score = int(float(cleaned))
    if score < 1 or score > 5:
        raise ValueError(f"Score must be between 1 and 5, got {raw_value!r}")
    return score


def parse_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def split_blocks(text: str) -> list[str]:
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []
    return [block.strip() for block in re.split(r"\n\s*\n+", cleaned) if block.strip()]


def parse_feedback_text(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for block_index, block in enumerate(split_blocks(text), start=1):
        parsed: dict[str, Any] = {"block_index": block_index}

        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped == "---" or stripped.startswith("#"):
                continue

            if "=" not in stripped:
                raise ValueError(f"Expected KEY=VALUE line in block {block_index}, got: {line!r}")

            raw_key, raw_value = stripped.split("=", 1)
            key = normalize_key(raw_key)
            value = raw_value.strip()

            if key in SCORE_FIELDS:
                parsed[key.lower()] = parse_score(value)
            elif key in LIST_FIELDS:
                parsed[key.lower()] = parse_list(value)
            elif key in TEXT_FIELDS:
                parsed[key.lower()] = value
            else:
                parsed[key.lower()] = value

        if "clip" not in parsed:
            raise ValueError(f"Missing CLIP in block {block_index}")
        if "verdict" not in parsed:
            raise ValueError(f"Missing VERDICT in block {block_index}")

        parsed["verdict"] = parsed["verdict"].strip().lower()
        entries.append(parsed)

    return entries


def average(entries: list[dict[str, Any]], field: str) -> float | None:
    values = [entry[field] for entry in entries if field in entry]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def build_recommendations(entries: list[dict[str, Any]]) -> list[str]:
    verdicts = Counter(entry["verdict"] for entry in entries)
    issues = Counter(issue for entry in entries for issue in entry.get("issues", []))
    recommendations: list[str] = []

    avg_similarity = average(entries, "similarity")
    avg_naturalness = average(entries, "naturalness")
    avg_pace = average(entries, "pace")
    avg_robotic = average(entries, "robotic")
    avg_clarity = average(entries, "clarity")

    if avg_similarity is not None and avg_similarity < 3:
        recommendations.append(
            "Voice identity is weak overall. Use the Clone Fidelity preset, keep random sampling off, and prefer a cleaner 10-20 second reference clip."
        )
    if (avg_pace is not None and avg_pace < 3) or issues["too_fast"] > 0:
        recommendations.append(
            "Pace sounds rushed. Add punctuation to the script, split long lines into shorter sentences, and tune with Clone Fidelity before moving back to more expressive settings."
        )
    if (avg_robotic is not None and avg_robotic >= 4) or (avg_naturalness is not None and avg_naturalness < 3):
        recommendations.append(
            "Naturalness is the main problem. Compare the same line with the Clone Fidelity preset, and avoid stacking strong emotion settings while checking base voice quality."
        )
    if (avg_clarity is not None and avg_clarity < 3) or issues["muffled"] > 0 or issues["slurred"] > 0:
        recommendations.append(
            "Clarity is low. Try a cleaner mono reference clip, shorter prompts, and simpler sentence wording before changing advanced decoding knobs."
        )
    if verdicts["bad"] >= max(1, len(entries) // 2) and "weak_similarity" in issues:
        recommendations.append(
            "The reference clip likely needs work. Trim silence/background noise and retest before judging model settings alone."
        )

    if not recommendations:
        recommendations.append("No major pattern detected yet. Keep collecting a few more rated clips before changing settings again.")

    return recommendations


def summarize_feedback(entries: list[dict[str, Any]]) -> dict[str, Any]:
    verdict_counts = Counter(entry["verdict"] for entry in entries)
    issue_counts = Counter(issue for entry in entries for issue in entry.get("issues", []))
    action_counts = Counter(action for entry in entries for action in entry.get("action", []))

    return {
        "count": len(entries),
        "verdict_counts": dict(verdict_counts),
        "average_scores": {
            "similarity": average(entries, "similarity"),
            "naturalness": average(entries, "naturalness"),
            "pace": average(entries, "pace"),
            "robotic": average(entries, "robotic"),
            "clarity": average(entries, "clarity"),
            "emotion": average(entries, "emotion"),
        },
        "top_issues": issue_counts.most_common(10),
        "top_actions": action_counts.most_common(10),
        "recommendations": build_recommendations(entries),
        "entries": entries,
    }


def main() -> int:
    args = parse_args()
    feedback_path = Path(args.feedback_file)
    feedback_text = feedback_path.read_text(encoding="utf-8")

    entries = parse_feedback_text(feedback_text)
    summary = summarize_feedback(entries)
    output_text = json.dumps(summary, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote parsed feedback to {output_path}")
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
