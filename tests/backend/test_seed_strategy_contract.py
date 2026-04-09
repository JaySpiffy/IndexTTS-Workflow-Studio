import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.api.core.conversation_manager import ConversationManager
from backend.api.services.conversation_service import ConversationService


class RecordingTTSCore:
    def __init__(self):
        self.tts = self
        self.calls = []

    def infer(self, *args, **kwargs):
        self.calls.append({
            "args": args,
            "kwargs": kwargs,
        })
        return args[2]

    def release_unused_memory(self, clear_prompt_cache=True):
        return True


class SeedStrategyContractTests(unittest.TestCase):
    def test_fixed_base_reused_list_reuses_same_seed_slots_for_each_line(self):
        tts_core = RecordingTTSCore()
        manager = ConversationManager(tts_core, SimpleNamespace())

        parsed_script = [
            {"speaker_filename": "speaker_a.wav", "text": "Line one.", "line_number": 0},
            {"speaker_filename": "speaker_b.wav", "text": "Line two.", "line_number": 1},
        ]

        with patch("backend.api.core.file_utils.prepare_temp_dir", return_value=True), \
             patch("backend.api.core.audio_processing.speaker_similarity_model", object()), \
             patch(
                 "backend.api.core.audio_processing.analyze_speaker_similarity_with_quality",
                 return_value={"similarity": 0.95, "robotic_score": 0.05, "quality_score": 0.91},
             ), \
             patch.object(ConversationManager, "cleanup_memory", return_value=True):
            updates = list(
                manager.generate_conversation(
                    parsed_script=parsed_script,
                    versions_per_line=2,
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
                    seed_strategy="fixed_base_reused_list",
                    fixed_base_seed=1234,
                    resolved_base_seed=1234,
                    reused_seed_list=[1234, 1235],
                )
            )

        self.assertEqual(
            [call["kwargs"]["seed"] for call in tts_core.calls],
            [1234, 1235, 1234, 1235],
        )
        line_results = updates[-1][3]
        self.assertEqual(line_results[0]["versions"][0]["seed"], 1234)
        self.assertEqual(line_results[1]["versions"][1]["seed"], 1235)
        self.assertEqual(line_results[0]["versions"][0]["seed_origin"], "initial")

    def test_fixed_base_sequential_offsets_seeds_by_line(self):
        tts_core = RecordingTTSCore()
        manager = ConversationManager(tts_core, SimpleNamespace())

        parsed_script = [
            {"speaker_filename": "speaker_a.wav", "text": "Line one.", "line_number": 0},
            {"speaker_filename": "speaker_b.wav", "text": "Line two.", "line_number": 1},
        ]

        with patch("backend.api.core.file_utils.prepare_temp_dir", return_value=True), \
             patch("backend.api.core.audio_processing.speaker_similarity_model", object()), \
             patch(
                 "backend.api.core.audio_processing.analyze_speaker_similarity_with_quality",
                 return_value={"similarity": 0.95, "robotic_score": 0.05, "quality_score": 0.91},
             ), \
             patch.object(ConversationManager, "cleanup_memory", return_value=True):
            list(
                manager.generate_conversation(
                    parsed_script=parsed_script,
                    versions_per_line=2,
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
                    seed_strategy="fixed_base_sequential",
                    fixed_base_seed=1234,
                    resolved_base_seed=1234,
                    reused_seed_list=[],
                )
            )

        self.assertEqual(
            [call["kwargs"]["seed"] for call in tts_core.calls],
            [1234, 1235, 1254, 1255],
        )

    def test_auto_regen_replaces_version_with_new_seed_metadata(self):
        tts_core = RecordingTTSCore()
        manager = ConversationManager(tts_core, SimpleNamespace())

        parsed_script = [
            {"speaker_filename": "speaker_a.wav", "text": "Line one.", "line_number": 0},
        ]

        with patch("backend.api.core.file_utils.prepare_temp_dir", return_value=True), \
             patch("backend.api.core.audio_processing.speaker_similarity_model", object()), \
             patch(
                 "backend.api.core.audio_processing.analyze_speaker_similarity_with_quality",
                 side_effect=[
                     {"similarity": 0.20, "robotic_score": 0.80, "quality_score": 0.18},
                     {"similarity": 0.91, "robotic_score": 0.05, "quality_score": 0.88},
                 ],
             ), \
             patch.object(ConversationManager, "_generate_unique_random_seed", return_value=42424242), \
             patch.object(ConversationManager, "cleanup_memory", return_value=True):
            updates = list(
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
                    seed_strategy="fixed_base_sequential",
                    fixed_base_seed=1234,
                    resolved_base_seed=1234,
                    reused_seed_list=[],
                )
            )

        self.assertEqual([call["kwargs"]["seed"] for call in tts_core.calls], [1234, 42424242])
        line_results = updates[-1][3]
        self.assertEqual(line_results[0]["versions"][0]["seed"], 42424242)
        self.assertEqual(line_results[0]["versions"][0]["seed_origin"], "auto_regen")

    def test_seed_runtime_resolves_reused_list_from_random_base(self):
        service = ConversationService(conversation_manager=None)

        with patch("backend.api.services.conversation_service.random.SystemRandom") as mock_system_random:
            mock_system_random.return_value.randrange.return_value = 777
            runtime = service._resolve_seed_runtime(
                seed_strategy="random_base_reused_list",
                fixed_base_seed=1234,
                versions_per_line=3,
            )

        self.assertEqual(runtime["resolved_base_seed"], 777)
        self.assertEqual(runtime["seed_strategy"], "random_base_reused_list")
        self.assertEqual(len(runtime["reused_seed_list"]), 3)
        self.assertEqual(runtime["reused_seed_list"], [984982403, 3231174163, 3962945185])


if __name__ == "__main__":
    unittest.main()
