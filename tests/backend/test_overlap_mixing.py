import tempfile
import unittest
import warnings
from pathlib import Path

from pydub import AudioSegment

from backend.api.core.audio_mixing import build_mix_timeline, mix_audio_files, parse_overlap_plan


class OverlapMixingTests(unittest.TestCase):
    def test_parse_overlap_plan_from_markdown_yaml_block(self):
        plan_text = """
# Example

```yaml
scene:
  title: "Argument Test"
  overlap_policy: explicit_only
  default_gap_ms: 250

lines:
  - id: L02
    start_mode: overlap_previous
    overlap_prev_ms: 180
    duck_prev_db: -5
    allow_overlap: true
```
"""
        parsed = parse_overlap_plan(plan_text, total_lines=3)
        self.assertEqual(parsed["scene"]["default_gap_ms"], 250)
        self.assertIn(1, parsed["lines"])
        self.assertEqual(parsed["lines"][1]["start_mode"], "overlap_previous")
        self.assertTrue(parsed["lines"][1]["allow_overlap"])

    def test_overlap_only_happens_when_explicitly_allowed(self):
        segments = [AudioSegment.silent(duration=1000), AudioSegment.silent(duration=1000)]
        no_overlap_timeline = build_mix_timeline(segments, None)
        self.assertEqual(no_overlap_timeline[1]["start_ms"], 1000)

        plan = parse_overlap_plan(
            """
scene:
  overlap_policy: explicit_only

lines:
  - id: L02
    start_mode: overlap_previous
    overlap_prev_ms: 300
    allow_overlap: true
""",
            total_lines=2,
        )
        overlap_timeline = build_mix_timeline(segments, plan)
        self.assertEqual(overlap_timeline[1]["start_ms"], 700)
        self.assertTrue(overlap_timeline[1]["overlap_applied"])

        disabled_plan = parse_overlap_plan(
            """
lines:
  - id: L02
    start_mode: overlap_previous
    overlap_prev_ms: 300
    allow_overlap: false
""",
            total_lines=2,
        )
        disabled_timeline = build_mix_timeline(segments, disabled_plan)
        self.assertEqual(disabled_timeline[1]["start_ms"], 1000)
        self.assertFalse(disabled_timeline[1]["overlap_applied"])

    def test_mix_audio_files_shortens_total_duration_when_overlap_is_used(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first = temp_path / "first.wav"
            second = temp_path / "second.wav"
            output = temp_path / "mixed.wav"

            first_export = AudioSegment.silent(duration=1000).export(first, format="wav")
            second_export = AudioSegment.silent(duration=1000).export(second, format="wav")
            first_export.close()
            second_export.close()

            result = mix_audio_files([str(first), str(second)], str(output))
            self.assertTrue(result["success"])
            self.assertEqual(result["duration_ms"], 2000)

            overlapped = mix_audio_files(
                [str(first), str(second)],
                str(output),
                overlap_plan_text="""
scene:
  overlap_policy: explicit_only

lines:
  - id: L02
    start_mode: overlap_previous
    overlap_prev_ms: 300
    allow_overlap: true
""",
            )
            self.assertTrue(overlapped["success"])
            self.assertEqual(overlapped["duration_ms"], 1700)


if __name__ == "__main__":
    unittest.main()
