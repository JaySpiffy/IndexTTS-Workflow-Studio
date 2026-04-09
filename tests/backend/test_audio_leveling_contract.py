import tempfile
import unittest
from pathlib import Path

from pydub.generators import Sine

from backend.api.core.audio_mixing import (
    _apply_final_peak_protection,
    _match_segment_loudness,
    mix_audio_files_at_positions,
)


def build_tone(gain_db: float, duration_ms: int = 800):
    return Sine(440).to_audio_segment(duration=duration_ms).apply_gain(gain_db)


class AudioLevelingContractTests(unittest.TestCase):
    def test_match_segment_loudness_moves_clip_toward_target(self):
        quiet_segment = build_tone(-24)

        adjusted, metadata = _match_segment_loudness(
            quiet_segment,
            target_level_dbfs=-18.0,
            peak_limit_dbfs=-1.0,
        )

        self.assertGreater(adjusted.dBFS, quiet_segment.dBFS)
        self.assertAlmostEqual(adjusted.dBFS, -18.0, delta=1.2)
        self.assertLessEqual(adjusted.max_dBFS, -0.95)
        self.assertGreater(metadata["applied_gain_db"], 0.0)

    def test_final_peak_protection_trims_hot_mix(self):
        hot_segment = build_tone(-0.1)

        adjusted, metadata = _apply_final_peak_protection(
            hot_segment,
            peak_limit_dbfs=-1.0,
        )

        self.assertLess(adjusted.max_dBFS, hot_segment.max_dBFS)
        self.assertLessEqual(adjusted.max_dBFS, -0.95)
        self.assertLess(metadata["applied_gain_db"], 0.0)

    def test_timeline_mix_reports_segment_leveling_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            quiet_path = temp_path / "quiet.wav"
            loud_path = temp_path / "loud.wav"
            output_path = temp_path / "mixed.wav"

            build_tone(-24).export(quiet_path, format="wav").close()
            build_tone(-8).export(loud_path, format="wav").close()

            result = mix_audio_files_at_positions(
                placements=[
                    {"audio_path": str(quiet_path), "start_ms": 0, "volume": 1.0, "track_id": "a", "segment_id": "one"},
                    {"audio_path": str(loud_path), "start_ms": 900, "volume": 1.0, "track_id": "a", "segment_id": "two"},
                ],
                output_path=str(output_path),
                normalize_segments=True,
                target_level_dbfs=-18.0,
                peak_limit_dbfs=-1.0,
                normalize_final_mix=True,
            )

            self.assertTrue(output_path.is_file())
            self.assertTrue(result["normalization"]["normalize_segments"])
            self.assertEqual(len(result["normalization"]["segment_adjustments"]), 2)
            applied_gains = [entry["applied_gain_db"] for entry in result["normalization"]["segment_adjustments"]]
            self.assertTrue(any(gain > 0 for gain in applied_gains))
            self.assertTrue(any(gain < 0 for gain in applied_gains))


if __name__ == "__main__":
    unittest.main()
