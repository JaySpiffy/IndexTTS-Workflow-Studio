import asyncio
import unittest
from copy import deepcopy

from backend.api.routers.conversation_results import regenerate_line_background
from backend.api.services.conversation_service import ConversationService


LINE_EMOTION_VECTOR = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]


def build_generation_params():
    return {
        "similarity_threshold": 0.60,
        "robotic_threshold": 0.70,
        "auto_regen_attempts": 1,
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
        "versions_per_line": 3,
    }


class RecordingConversationManager:
    def __init__(self, response_batches):
        self.response_batches = [deepcopy(batch) for batch in response_batches]
        self.regenerate_calls = []

    def regenerate_line(self, line_number, line_data, regen_count, **kwargs):
        self.regenerate_calls.append(
            {
                "line_number": line_number,
                "line_data": deepcopy(line_data),
                "regen_count": regen_count,
                "kwargs": deepcopy(kwargs),
            }
        )
        response_index = len(self.regenerate_calls) - 1
        versions = self.response_batches[response_index]
        yield ("Regeneration complete", "", 100, versions)


class ReviewRegenerationContractTests(unittest.TestCase):
    def test_replace_all_regeneration_uses_edited_text_and_reselects_best_version(self):
        manager = RecordingConversationManager(
            [
                [
                    {
                        "audio_path": "temp_conversation_segments/new_v1.wav",
                        "similarity_score": 0.71,
                        "robotic_score": 0.10,
                        "quality_score": 0.61,
                        "speaker_filename": "speaker.wav",
                        "text": "Updated hello",
                    },
                    {
                        "audio_path": "temp_conversation_segments/new_v2.wav",
                        "similarity_score": 0.82,
                        "robotic_score": 0.09,
                        "quality_score": 0.79,
                        "speaker_filename": "speaker.wav",
                        "text": "Updated hello",
                    },
                ]
            ]
        )
        service = ConversationService(conversation_manager=manager)
        service.active_conversations["conversation-1"] = {
            "parsed_script": [
                {
                    "speaker_filename": "speaker.wav",
                    "text": "Hello world",
                    "line_number": 0,
                    "emotion_vectors": LINE_EMOTION_VECTOR,
                }
            ],
            "generation_params": build_generation_params(),
            "lines": [
                {
                    "line_number": 0,
                    "speaker_filename": "speaker.wav",
                    "text": "Hello world",
                    "versions": [
                        {
                            "audio_path": "temp_conversation_segments/original.wav",
                            "similarity_score": 0.65,
                            "robotic_score": 0.12,
                            "quality_score": 0.52,
                            "is_selected": True,
                        }
                    ],
                }
            ],
        }
        service.active_conversations["regen_conversation-1_0"] = {
            "status": "pending",
            "progress": 0.0,
            "current_step": "Initializing regeneration",
            "line_number": 0,
            "regen_count": 2,
            "conversation_id": "conversation-1",
            "mode": "replace_all",
            "edited_text": "Updated hello",
            "manual_similarity_threshold": None,
            "max_manual_attempts": None,
            "error": None,
            "start_time": 0.0,
            "end_time": None,
            "new_versions": [],
        }

        asyncio.run(
            regenerate_line_background(
                regen_task_id="regen_conversation-1_0",
                conversation_id="conversation-1",
                line_number=0,
                regen_count=2,
                line_data={"speaker_filename": "speaker.wav", "text": "Hello world"},
                conversation_service=service,
            )
        )

        self.assertEqual(len(manager.regenerate_calls), 1)
        self.assertEqual(manager.regenerate_calls[0]["line_data"]["text"], "Updated hello")
        self.assertEqual(service.active_conversations["conversation-1"]["parsed_script"][0]["text"], "Updated hello")
        self.assertEqual(service.active_conversations["conversation-1"]["lines"][0]["text"], "Updated hello")

        versions = service.active_conversations["conversation-1"]["lines"][0]["versions"]
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[1]["audio_filename"], "new_v2.wav")
        self.assertTrue(versions[1]["is_selected"])
        self.assertFalse(versions[0]["is_selected"])

    def test_replace_all_regeneration_keeps_best_available_selection_when_all_versions_fail_quality_gate(self):
        manager = RecordingConversationManager(
            [
                [
                    {
                        "audio_path": "temp_conversation_segments/fail_v1.wav",
                        "similarity_score": 0.41,
                        "robotic_score": 0.25,
                        "quality_score": 0.36,
                        "speaker_filename": "speaker.wav",
                        "text": "Updated hello",
                    },
                    {
                        "audio_path": "temp_conversation_segments/fail_v2.wav",
                        "similarity_score": 0.53,
                        "robotic_score": 0.25,
                        "quality_score": 0.44,
                        "speaker_filename": "speaker.wav",
                        "text": "Updated hello",
                    },
                ]
            ]
        )
        service = ConversationService(conversation_manager=manager)
        service.active_conversations["conversation-1"] = {
            "parsed_script": [
                {
                    "speaker_filename": "speaker.wav",
                    "text": "Hello world",
                    "line_number": 0,
                    "emotion_vectors": LINE_EMOTION_VECTOR,
                }
            ],
            "generation_params": build_generation_params(),
            "lines": [
                {
                    "line_number": 0,
                    "speaker_filename": "speaker.wav",
                    "text": "Hello world",
                    "versions": [
                        {
                            "audio_path": "temp_conversation_segments/original.wav",
                            "similarity_score": 0.65,
                            "robotic_score": 0.12,
                            "quality_score": 0.52,
                            "is_selected": True,
                        }
                    ],
                }
            ],
        }
        service.active_conversations["regen_conversation-1_0"] = {
            "status": "pending",
            "progress": 0.0,
            "current_step": "Initializing regeneration",
            "line_number": 0,
            "regen_count": 2,
            "conversation_id": "conversation-1",
            "mode": "replace_all",
            "edited_text": "Updated hello",
            "manual_similarity_threshold": None,
            "max_manual_attempts": None,
            "error": None,
            "start_time": 0.0,
            "end_time": None,
            "new_versions": [],
        }

        asyncio.run(
            regenerate_line_background(
                regen_task_id="regen_conversation-1_0",
                conversation_id="conversation-1",
                line_number=0,
                regen_count=2,
                line_data={"speaker_filename": "speaker.wav", "text": "Hello world"},
                conversation_service=service,
            )
        )

        versions = service.active_conversations["conversation-1"]["lines"][0]["versions"]
        self.assertEqual(len(versions), 2)
        self.assertFalse(versions[0]["is_selected"])
        self.assertTrue(versions[1]["is_selected"])

    def test_threshold_regeneration_only_replaces_low_similarity_slots(self):
        manager = RecordingConversationManager(
            [
                [
                    {
                        "audio_path": "temp_conversation_segments/slot1_new.wav",
                        "similarity_score": 0.77,
                        "robotic_score": 0.10,
                        "quality_score": 0.72,
                        "speaker_filename": "speaker.wav",
                        "text": "Adjusted line",
                    },
                    {
                        "audio_path": "temp_conversation_segments/slot2_candidate.wav",
                        "similarity_score": 0.61,
                        "robotic_score": 0.18,
                        "quality_score": 0.20,
                        "speaker_filename": "speaker.wav",
                        "text": "Adjusted line",
                    },
                ]
            ]
        )
        service = ConversationService(conversation_manager=manager)
        service.active_conversations["conversation-1"] = {
            "parsed_script": [
                {
                    "speaker_filename": "speaker.wav",
                    "text": "Original line",
                    "line_number": 0,
                    "emotion_vectors": LINE_EMOTION_VECTOR,
                }
            ],
            "generation_params": build_generation_params(),
            "lines": [
                {
                    "line_number": 0,
                    "speaker_filename": "speaker.wav",
                    "text": "Original line",
                    "versions": [
                        {
                            "audio_path": "temp_conversation_segments/slot0_old.wav",
                            "audio_filename": "slot0_old.wav",
                            "similarity_score": 0.88,
                            "robotic_score": 0.05,
                            "quality_score": 0.81,
                            "is_selected": False,
                        },
                        {
                            "audio_path": "temp_conversation_segments/slot1_old.wav",
                            "audio_filename": "slot1_old.wav",
                            "similarity_score": 0.35,
                            "robotic_score": 0.10,
                            "quality_score": 0.40,
                            "is_selected": True,
                        },
                        {
                            "audio_path": "temp_conversation_segments/slot2_old.wav",
                            "audio_filename": "slot2_old.wav",
                            "similarity_score": 0.25,
                            "robotic_score": 0.09,
                            "quality_score": 0.30,
                            "is_selected": False,
                        },
                    ],
                }
            ],
        }
        service.active_conversations["regen_conversation-1_0"] = {
            "status": "pending",
            "progress": 0.0,
            "current_step": "Initializing regeneration",
            "line_number": 0,
            "regen_count": 3,
            "conversation_id": "conversation-1",
            "mode": "below_threshold",
            "edited_text": "Adjusted line",
            "manual_similarity_threshold": 0.60,
            "max_manual_attempts": 3,
            "error": None,
            "start_time": 0.0,
            "end_time": None,
            "new_versions": [],
        }

        asyncio.run(
            regenerate_line_background(
                regen_task_id="regen_conversation-1_0",
                conversation_id="conversation-1",
                line_number=0,
                regen_count=3,
                line_data={"speaker_filename": "speaker.wav", "text": "Original line"},
                conversation_service=service,
            )
        )

        self.assertEqual(len(manager.regenerate_calls), 1)
        self.assertEqual(manager.regenerate_calls[0]["regen_count"], 2)
        self.assertEqual(manager.regenerate_calls[0]["line_data"]["text"], "Adjusted line")
        self.assertEqual(manager.regenerate_calls[0]["kwargs"]["similarity_threshold"], 0.60)
        self.assertEqual(manager.regenerate_calls[0]["kwargs"]["auto_regen_attempts"], 2)

        line = service.active_conversations["conversation-1"]["lines"][0]
        versions = line["versions"]
        self.assertEqual(line["text"], "Adjusted line")
        self.assertEqual(service.active_conversations["conversation-1"]["parsed_script"][0]["text"], "Adjusted line")
        self.assertEqual(len(versions), 3)
        self.assertEqual(versions[0]["audio_path"], "temp_conversation_segments/slot0_old.wav")
        self.assertEqual(versions[1]["audio_path"], "temp_conversation_segments/slot1_new.wav")
        self.assertEqual(versions[2]["audio_path"], "temp_conversation_segments/slot2_old.wav")
        self.assertTrue(versions[1]["is_selected"])
        self.assertFalse(versions[0]["is_selected"])
        self.assertFalse(versions[2]["is_selected"])


if __name__ == "__main__":
    unittest.main()
