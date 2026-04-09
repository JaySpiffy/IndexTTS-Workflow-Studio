import tempfile
import unittest
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from backend.api.core.source_clip_prep import analyze_source_clip, prepare_source_clip


def build_stereo_clip() -> AudioSegment:
    tone = Sine(440).to_audio_segment(duration=1600).apply_gain(-18)
    stereo = AudioSegment.from_mono_audiosegments(tone, tone)
    return AudioSegment.silent(duration=600) + stereo + AudioSegment.silent(duration=900)


def build_long_clip_with_strong_middle() -> AudioSegment:
    quiet_intro = Sine(220).to_audio_segment(duration=12000).apply_gain(-32)
    strong_middle = Sine(440).to_audio_segment(duration=18000).apply_gain(-8)
    quiet_tail = Sine(330).to_audio_segment(duration=12000).apply_gain(-30)
    return quiet_intro + strong_middle + quiet_tail


class SourceClipPrepContractTests(unittest.TestCase):
    def test_diagnostics_flag_common_clone_issues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "messy_source.wav"
            export_handle = build_stereo_clip().export(source_path, format="wav")
            export_handle.close()

            diagnostics = analyze_source_clip(source_path)

            self.assertEqual(diagnostics["channels"], 2)
            self.assertGreater(diagnostics["leading_silence_ms"], 300)
            self.assertGreater(diagnostics["trailing_silence_ms"], 500)
            joined_recommendations = " ".join(diagnostics["recommendations"]).lower()
            self.assertIn("mono", joined_recommendations)
            self.assertIn("trim", joined_recommendations)
            self.assertIn("suggested_prep", diagnostics)
            self.assertGreater(diagnostics["suggested_prep"]["start_time"], 0.0)
            self.assertIsNotNone(diagnostics["suggested_prep"]["end_time"])
            self.assertTrue(diagnostics["suggested_prep"]["convert_to_mono"])

    def test_prepare_source_clip_trims_and_converts_to_mono(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "messy_source.wav"
            output_path = temp_path / "clean_speaker.wav"
            export_handle = build_stereo_clip().export(source_path, format="wav")
            export_handle.close()

            result = prepare_source_clip(
                source_path,
                output_path,
                start_time=0.6,
                end_time=2.2,
                convert_to_mono=True,
                normalize_audio=True,
                target_peak_dbfs=-1.0,
                use_noise_reduction=False,
                use_vocal_separation=False,
            )

            self.assertTrue(output_path.is_file())
            self.assertEqual(result["after"]["channels"], 1)
            self.assertLess(result["after"]["duration_seconds"], result["before"]["duration_seconds"])
            self.assertLessEqual(result["after"]["peak_dbfs"], -0.8)
            self.assertTrue(any("mono" in note.lower() for note in result["processing_notes"]))
            self.assertGreaterEqual(
                result["after"]["clone_readiness_score"],
                result["before"]["clone_readiness_score"],
            )

    def test_diagnostics_pick_stronger_middle_window_for_long_clips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "long_middle.wav"
            export_handle = build_long_clip_with_strong_middle().export(source_path, format="wav")
            export_handle.close()

            diagnostics = analyze_source_clip(source_path)
            suggested = diagnostics["suggested_prep"]

            self.assertGreaterEqual(suggested["start_time"], 10.0)
            self.assertLessEqual(suggested["start_time"], 14.0)
            self.assertIsNotNone(suggested["end_time"])
            self.assertLessEqual((suggested["end_time"] - suggested["start_time"]), 20.0)


if __name__ == "__main__":
    unittest.main()
