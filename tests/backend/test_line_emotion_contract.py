import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.api.core.conversation_manager import ConversationManager
from backend.api.models import ConversationGenerationRequest
from backend.api.routers.conversation_results import regenerate_line_background
from backend.api.services.conversation_service import ConversationService


LINE_EMOTION_VECTOR = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]


class FakeTTS:
    def __init__(self):
        self.calls = []

    def infer(self, speaker_prompt, text, output_path, **kwargs):
        self.calls.append(
            {
                "speaker_prompt": speaker_prompt,
                "text": text,
                "output_path": output_path,
                **kwargs,
            }
        )
        return output_path


class RecordingConversationManager:
    def __init__(self):
        self.generate_calls = []
        self.regenerate_calls = []

    def generate_conversation(self, parsed_script, **kwargs):
        self.generate_calls.append({"parsed_script": parsed_script, "kwargs": kwargs})
        yield (
            "Generation complete",
            "",
            100,
            [
                {
                    "line_index": 0,
                    "speaker_filename": parsed_script[0]["speaker_filename"],
                    "text": parsed_script[0]["text"],
                    "versions": [
                        {
                            "audio_path": "temp_conversation_segments/line000_spk-speaker_v01.wav",
                            "similarity_score": 0.99,
                            "robotic_score": 0.01,
                            "quality_score": 0.99,
                        }
                    ],
                }
            ],
            None,
        )

    def regenerate_line(self, line_number, line_data, regen_count, **kwargs):
        self.regenerate_calls.append(
            {
                "line_number": line_number,
                "line_data": line_data,
                "regen_count": regen_count,
                "kwargs": kwargs,
            }
        )
        yield (
            "Regeneration complete",
            "",
            100,
            [
                {
                    "audio_path": f"temp_conversation_segments/line{line_number:03d}_regen01.wav",
                    "similarity_score": 0.99,
                    "robotic_score": 0.01,
                    "quality_score": 0.99,
                    "speaker_filename": line_data["speaker_filename"],
                    "text": line_data["text"],
                }
            ],
        )


class LineEmotionContractTests(unittest.TestCase):
    def test_conversation_request_accepts_frontend_emo_vector_alias(self):
        request = ConversationGenerationRequest(
            script={
                "title": "Alias test",
                "lines": [
                    {
                        "speaker_filename": "speaker.wav",
                        "text": "Hello world",
                        "line_number": 0,
                        "emo_vector": LINE_EMOTION_VECTOR,
                    }
                ],
            }
        )

        self.assertEqual(request.script.lines[0].emotion_vectors, LINE_EMOTION_VECTOR)

    def test_conversation_service_preserves_line_emotion_metadata(self):
        manager = RecordingConversationManager()
        service = ConversationService(conversation_manager=manager)
        parsed_script = [
            {
                "speaker_filename": "speaker.wav",
                "text": "Hello world",
                "line_number": 0,
                "emotion_vectors": LINE_EMOTION_VECTOR,
            }
        ]

        with patch("backend.api.core.file_utils.validate_speaker_files", return_value=[]):
            task_info = service.start_conversation_generation(parsed_script=parsed_script)
            asyncio.run(service.generate_conversation_async(task_info["conversation_id"]))

        self.assertEqual(len(manager.generate_calls), 1)
        self.assertEqual(
            manager.generate_calls[0]["parsed_script"][0]["emotion_vectors"],
            LINE_EMOTION_VECTOR,
        )

    def test_conversation_manager_uses_line_emotion_vectors_for_generation(self):
        fake_tts = FakeTTS()
        tts_core = SimpleNamespace(tts=fake_tts)
        manager = ConversationManager(tts_core=tts_core, cmd_args=SimpleNamespace(verbose=False))
        parsed_script = [
            {
                "speaker_filename": "speaker.wav",
                "text": "Hello world",
                "line_number": 0,
                "emotion_vectors": LINE_EMOTION_VECTOR,
            }
        ]

        with patch("backend.api.core.file_utils.prepare_temp_dir", return_value=True), patch(
            "backend.api.core.audio_processing.analyze_speaker_similarity_with_quality",
            return_value={"similarity": 0.99, "robotic_score": 0.01, "quality_score": 0.99},
        ):
            list(
                manager.generate_conversation(
                    parsed_script=parsed_script,
                    versions_per_line=1,
                    similarity_threshold=0.6,
                    robotic_threshold=0.7,
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

        self.assertEqual(len(fake_tts.calls), 1)
        self.assertEqual(fake_tts.calls[0]["emo_vector"], LINE_EMOTION_VECTOR)
        self.assertFalse(fake_tts.calls[0]["use_emo_text"])

    def test_regeneration_reuses_line_emotion_metadata_from_parsed_script(self):
        manager = RecordingConversationManager()

        class FakeConversationService:
            def __init__(self):
                self.conversation_manager = manager
                self.active_conversations = {
                    "conversation-1": {
                        "parsed_script": [
                            {
                                "speaker_filename": "speaker.wav",
                                "text": "Hello world",
                                "line_number": 0,
                                "emotion_vectors": LINE_EMOTION_VECTOR,
                            }
                        ],
                        "generation_params": {
                            "similarity_threshold": 0.6,
                            "robotic_threshold": 0.7,
                            "auto_regen_attempts": 0,
                            "emotion_control_method": "from_speaker",
                            "emotion_reference_filename": None,
                            "emotion_weight": 1.0,
                            "emotion_vectors": [],
                            "emotion_text": None,
                            "use_random_sampling": False,
                            "do_sample": True,
                            "top_p": 0.8,
                            "top_k": 30,
                            "temperature": 0.8,
                            "length_penalty": 0.0,
                            "num_beams": 3,
                            "repetition_penalty": 10.0,
                            "max_mel_tokens": 1500,
                            "max_text_tokens_per_segment": 120,
                        },
                        "lines": [
                            {
                                "speaker_filename": "speaker.wav",
                                "text": "Hello world",
                                "versions": [],
                            }
                        ],
                    },
                    "regen-conversation-1-0": {
                        "status": "pending",
                        "progress": 0.0,
                        "current_step": "Initializing regeneration",
                        "line_number": 0,
                        "regen_count": 1,
                        "conversation_id": "conversation-1",
                        "error": None,
                        "start_time": 0.0,
                        "end_time": None,
                        "new_versions": [],
                    },
                }

        service = FakeConversationService()

        asyncio.run(
            regenerate_line_background(
                regen_task_id="regen-conversation-1-0",
                conversation_id="conversation-1",
                line_number=0,
                regen_count=1,
                line_data={"speaker_filename": "speaker.wav", "text": "Hello world"},
                conversation_service=service,
            )
        )

        self.assertEqual(len(manager.regenerate_calls), 1)
        self.assertEqual(
            manager.regenerate_calls[0]["line_data"]["emotion_vectors"],
            LINE_EMOTION_VECTOR,
        )


if __name__ == "__main__":
    unittest.main()
