import tempfile
import unittest
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from backend.api.core.audio_mixing import mix_audio_files, mix_audio_files_at_positions


def build_padded_tone(gain_db: float, *, tone_ms: int = 700, leading_ms: int = 300, trailing_ms: int = 450) -> AudioSegment:
    tone = Sine(440).to_audio_segment(duration=tone_ms).apply_gain(gain_db)
    return AudioSegment.silent(duration=leading_ms) + tone + AudioSegment.silent(duration=trailing_ms)


class ExportFinishingContractTests(unittest.TestCase):
    def test_conversation_mix_trims_final_silence_and_exports_mp3(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_path = temp_path / "first.wav"
            second_path = temp_path / "second.wav"
            output_path = temp_path / "conversation_mix.mp3"

            build_padded_tone(-14).export(first_path, format="wav").close()
            build_padded_tone(-12, leading_ms=200, trailing_ms=500).export(second_path, format="wav").close()

            result = mix_audio_files(
                [str(first_path), str(second_path)],
                str(output_path),
                output_format="mp3",
                output_bitrate_kbps=128,
                normalize_segments=False,
                normalize_final_mix=False,
                trim_leading_silence=True,
                trim_trailing_silence=True,
                trim_silence_threshold_dbfs=-40.0,
                trim_min_silence_len_ms=100,
                fade_in_ms=0,
                fade_out_ms=80,
            )

            self.assertTrue(output_path.is_file())
            self.assertEqual(result["output_format"], "mp3")
            self.assertEqual(result["finishing"]["output_bitrate_kbps"], 128)
            self.assertGreater(result["finishing"]["silence_trim"]["leading_trim_ms"], 0)
            self.assertGreater(result["finishing"]["silence_trim"]["trailing_trim_ms"], 0)

            exported = AudioSegment.from_file(output_path, format="mp3")
            self.assertLess(len(exported), 2 * len(build_padded_tone(-14)))

    def test_timeline_mix_reports_fade_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            segment_path = temp_path / "segment.wav"
            output_path = temp_path / "timeline_mix.wav"

            build_padded_tone(-10, leading_ms=0, trailing_ms=0).export(segment_path, format="wav").close()

            result = mix_audio_files_at_positions(
                placements=[
                    {
                        "audio_path": str(segment_path),
                        "start_ms": 0,
                        "volume": 1.0,
                        "track_id": "track-1",
                        "segment_id": "segment-1",
                    }
                ],
                output_path=str(output_path),
                output_format="wav",
                normalize_segments=False,
                normalize_final_mix=False,
                trim_leading_silence=False,
                trim_trailing_silence=False,
                fade_in_ms=120,
                fade_out_ms=180,
            )

            self.assertTrue(output_path.is_file())
            self.assertEqual(result["output_format"], "wav")
            self.assertTrue(result["finishing"]["fades"]["applied"])
            self.assertEqual(result["finishing"]["fades"]["fade_in_ms"], 120)
            self.assertEqual(result["finishing"]["fades"]["fade_out_ms"], 180)

    def test_timeline_mix_exports_ogg_without_mp3_bitrate_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            segment_path = temp_path / "segment.wav"
            output_path = temp_path / "timeline_mix.ogg"

            build_padded_tone(-10, leading_ms=0, trailing_ms=0).export(segment_path, format="wav").close()

            result = mix_audio_files_at_positions(
                placements=[
                    {
                        "audio_path": str(segment_path),
                        "start_ms": 0,
                        "volume": 1.0,
                        "track_id": "track-1",
                        "segment_id": "segment-1",
                    }
                ],
                output_path=str(output_path),
                output_format="ogg",
                output_bitrate_kbps=192,
                normalize_segments=False,
                normalize_final_mix=False,
                trim_leading_silence=False,
                trim_trailing_silence=False,
            )

            self.assertTrue(output_path.is_file())
            self.assertEqual(result["output_format"], "ogg")
            self.assertIsNone(result["finishing"]["output_bitrate_kbps"])


if __name__ == "__main__":
    unittest.main()
