import shutil
import tempfile
import unittest
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from backend.api.models import ConversationGenerationRequest
from backend.api.core.audio_mixing import build_mix_timeline
from backend.api.core.pacing import (
    apply_delivery_rate_to_file,
    build_speaker_pacing_map,
    resolve_speaker_delivery_rate,
)


class PacingContractTests(unittest.TestCase):
    @staticmethod
    def _duration_ms(audio_path: Path) -> int:
        with open(audio_path, "rb") as audio_file:
            return len(AudioSegment.from_file(audio_file, format="wav"))

    def test_generation_request_accepts_named_pacing_preset(self):
        request = ConversationGenerationRequest.model_validate({
            "script": {
                "title": "Pacing preset test",
                "lines": [
                    {
                        "speaker_filename": "Pr.D.Trump.wav",
                        "text": "Who are you?",
                    }
                ],
            },
            "pacing_preset": "argument",
        })

        self.assertEqual(request.pacing_preset.value, "argument")

    def test_speaker_pacing_map_resolves_by_filename_and_stem(self):
        pacing_map = build_speaker_pacing_map([
            {"speaker_filename": "Pr.D.Trump.wav", "delivery_rate": 0.92},
            {"speaker_filename": "JoeRogan.wav", "delivery_rate": 1.08},
        ])

        self.assertAlmostEqual(resolve_speaker_delivery_rate("Pr.D.Trump.wav", pacing_map), 0.92, places=2)
        self.assertAlmostEqual(resolve_speaker_delivery_rate("Pr.D.Trump", pacing_map), 0.92, places=2)
        self.assertAlmostEqual(resolve_speaker_delivery_rate("JoeRogan", pacing_map), 1.08, places=2)
        self.assertAlmostEqual(resolve_speaker_delivery_rate("Unknown.wav", pacing_map), 1.0, places=2)

    def test_build_mix_timeline_respects_previous_line_punctuation(self):
        segments = [
            AudioSegment.silent(duration=400),
            AudioSegment.silent(duration=500),
        ]

        timeline = build_mix_timeline(
            segments,
            line_texts=["Who are you?", "Answer me."],
            scene_pacing_profile="relaxed",
            scene_gap_ms=220,
            respect_punctuation_pauses=True,
        )

        self.assertEqual(timeline[0]["start_ms"], 0)
        self.assertEqual(timeline[1]["punctuation_pause_ms"], 150)
        self.assertEqual(timeline[1]["start_ms"], 770)

    def test_apply_delivery_rate_to_file_changes_clip_duration(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg is required for delivery-rate shaping")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "pace-test.wav"

            exported = Sine(440).to_audio_segment(duration=1000).export(audio_path, format="wav")
            close_export = getattr(exported, "close", None)
            if callable(close_export):
                close_export()

            original_duration_ms = self._duration_ms(audio_path)
            result = apply_delivery_rate_to_file(str(audio_path), 1.10)
            paced_duration_ms = self._duration_ms(audio_path)

            self.assertTrue(result["applied"])
            self.assertAlmostEqual(result["delivery_rate"], 1.10, places=2)
            self.assertLess(paced_duration_ms, original_duration_ms - 40)


if __name__ == "__main__":
    unittest.main()
