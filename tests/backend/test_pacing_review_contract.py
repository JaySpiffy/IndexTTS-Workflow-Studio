import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.api.core.conversation_manager import ConversationManager
from backend.api.core.pacing import assess_line_pacing


def write_silent_wav(audio_path: Path, duration_seconds: float, sample_rate: int = 22050) -> None:
    frame_count = int(sample_rate * duration_seconds)
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


class WavWritingFakeTTS:
    def infer(self, speaker_prompt, text, output_path, **kwargs):
        write_silent_wav(Path(output_path), 2.4)
        return output_path


class PacingReviewContractTests(unittest.TestCase):
    def test_assess_line_pacing_flags_rushed_audio_lower_than_balanced_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            rushed_path = temp_path / "rushed.wav"
            balanced_path = temp_path / "balanced.wav"
            text = "This line should sound naturally paced, not rushed."

            write_silent_wav(rushed_path, 1.0)
            write_silent_wav(balanced_path, 3.3)

            rushed = assess_line_pacing(text, str(rushed_path))
            balanced = assess_line_pacing(text, str(balanced_path))

            self.assertEqual(rushed["pacing_label"], "too_fast")
            self.assertGreater(balanced["pacing_score"], rushed["pacing_score"])
            self.assertEqual(balanced["pacing_label"], "balanced")
            self.assertGreater(rushed["pacing_score"], 0.0)

    def test_assess_line_pacing_is_less_harsh_on_short_conversational_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            slow_path = temp_path / "slow-short-line.wav"
            fast_path = temp_path / "fast-short-line.wav"

            slow_text = "We are doing a final release smoke today."
            fast_text = "I just want to make sure the workflow really works."

            write_silent_wav(slow_path, 4.365)
            write_silent_wav(fast_path, 2.229)

            slow_result = assess_line_pacing(slow_text, str(slow_path))
            fast_result = assess_line_pacing(fast_text, str(fast_path))

            self.assertEqual(slow_result["pacing_label"], "slightly_slow")
            self.assertGreaterEqual(slow_result["pacing_score"], 0.65)
            self.assertEqual(fast_result["pacing_label"], "too_fast")
            self.assertGreaterEqual(fast_result["pacing_score"], 0.45)

    def test_conversation_generation_versions_include_pacing_review_fields(self):
        manager = ConversationManager(
            tts_core=SimpleNamespace(tts=WavWritingFakeTTS()),
            cmd_args=SimpleNamespace(verbose=False),
        )

        with patch("backend.api.core.file_utils.prepare_temp_dir", return_value=True), patch(
            "backend.api.core.audio_processing.analyze_speaker_similarity_with_quality",
            return_value={"similarity": 0.94, "robotic_score": 0.08, "quality_score": 0.87},
        ):
            updates = list(
                manager.generate_conversation(
                    parsed_script=[
                        {
                            "speaker_filename": "speaker.wav",
                            "text": "A natural paced review contract line.",
                            "line_number": 0,
                        }
                    ],
                    versions_per_line=1,
                    similarity_threshold=0.60,
                    robotic_threshold=0.70,
                    auto_regen_attempts=0,
                    emo_control_method=0,
                    emo_ref_path=None,
                    emo_weight=1.0,
                    emo_random=False,
                    vec1=0.0,
                    vec2=0.0,
                    vec3=0.0,
                    vec4=0.0,
                    vec5=0.0,
                    vec6=0.0,
                    vec7=0.0,
                    vec8=0.0,
                    emo_text=None,
                    do_sample_convo=False,
                    top_p_convo=0.8,
                    top_k_convo=30,
                    temperature_convo=0.8,
                    length_penalty_convo=0.0,
                    num_beams_convo=3,
                    repetition_penalty_convo=10.0,
                    max_mel_tokens_convo=1500,
                    max_text_tokens_per_segment_convo=120,
                    scene_pacing_profile="balanced",
                )
            )

        line_results = updates[-1][3]
        version = line_results[0]["versions"][0]

        self.assertIn("duration_seconds", version)
        self.assertIn("pacing_score", version)
        self.assertIn("pacing_label", version)
        self.assertIn("review_score", version)
        self.assertGreater(version["duration_seconds"], 0.0)
        self.assertGreater(version["review_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
