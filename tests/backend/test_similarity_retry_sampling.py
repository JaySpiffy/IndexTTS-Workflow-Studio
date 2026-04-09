import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.api.core.conversation_manager import ConversationManager


class RecordingTTSCore:
    def __init__(self):
        self.tts = self
        self.calls = []

    def infer(self, *args, **kwargs):
        self.calls.append({
            "args": args,
            "kwargs": kwargs,
        })

    def release_unused_memory(self, clear_prompt_cache=True):
        return True


class ConversationRetrySamplingTests(unittest.TestCase):
    def test_auto_regen_preserves_requested_random_sampling_flag(self):
        tts_core = RecordingTTSCore()
        manager = ConversationManager(tts_core, SimpleNamespace())

        parsed_script = [
            {
                "speaker_filename": "Pr.D.Trump.wav",
                "text": "This is a fidelity check.",
                "line_number": 0,
            }
        ]

        with patch("backend.api.core.file_utils.prepare_temp_dir", return_value=True), \
             patch("backend.api.core.audio_processing.speaker_similarity_model", object()), \
             patch(
                 "backend.api.core.audio_processing.analyze_speaker_similarity_with_quality",
                 side_effect=[
                     {"similarity": 0.20, "robotic_score": 0.10, "quality_score": 0.18},
                     {"similarity": 0.24, "robotic_score": 0.10, "quality_score": 0.22},
                 ],
             ), \
             patch.object(ConversationManager, "cleanup_memory", return_value=True):
            list(
                manager.generate_conversation(
                    parsed_script=parsed_script,
                    versions_per_line=1,
                    similarity_threshold=0.60,
                    robotic_threshold=0.70,
                    auto_regen_attempts=1,
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
                    vec8=0.8,
                    emo_text="",
                    do_sample_convo=True,
                    top_p_convo=0.8,
                    top_k_convo=30,
                    temperature_convo=0.8,
                    length_penalty_convo=0.0,
                    num_beams_convo=3,
                    repetition_penalty_convo=10.0,
                    max_mel_tokens_convo=1500,
                    max_text_tokens_per_segment_convo=120,
                )
            )

        self.assertEqual(len(tts_core.calls), 2)
        self.assertFalse(tts_core.calls[0]["kwargs"]["use_random"])
        self.assertFalse(tts_core.calls[1]["kwargs"]["use_random"])

    def test_manual_regeneration_preserves_requested_random_sampling_flag(self):
        tts_core = RecordingTTSCore()
        manager = ConversationManager(tts_core, SimpleNamespace())

        line_data = {
            "speaker_filename": "Pr.D.Trump.wav",
            "text": "This is a manual retry check.",
        }

        with patch("backend.api.core.file_utils.prepare_temp_dir", return_value=True), \
             patch("backend.api.core.audio_processing.speaker_similarity_model", object()), \
             patch(
                 "backend.api.core.audio_processing.analyze_speaker_similarity_with_quality",
                 side_effect=[
                     {"similarity": 0.20, "robotic_score": 0.10, "quality_score": 0.18},
                     {"similarity": 0.24, "robotic_score": 0.10, "quality_score": 0.22},
                 ],
             ), \
             patch.object(ConversationManager, "cleanup_memory", return_value=True):
            list(
                manager.regenerate_line(
                    line_index=0,
                    line_data=line_data,
                    num_versions=1,
                    similarity_threshold=0.60,
                    robotic_threshold=0.70,
                    auto_regen_attempts=1,
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
                    vec8=0.8,
                    emo_text="",
                    do_sample_convo=True,
                    top_p_convo=0.8,
                    top_k_convo=30,
                    temperature_convo=0.8,
                    length_penalty_convo=0.0,
                    num_beams_convo=3,
                    repetition_penalty_convo=10.0,
                    max_mel_tokens_convo=1500,
                    max_text_tokens_per_segment_convo=120,
                )
            )

        self.assertEqual(len(tts_core.calls), 2)
        self.assertFalse(tts_core.calls[0]["kwargs"]["use_random"])
        self.assertFalse(tts_core.calls[1]["kwargs"]["use_random"])


if __name__ == "__main__":
    unittest.main()
