import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.api.services.conversation_service import ConversationService


LINE_EMOTION_VECTOR = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]


class ProjectSaveLoadTests(unittest.TestCase):
    def test_save_and_load_round_trip_restores_conversation_snapshot(self):
        service = ConversationService(conversation_manager=None)
        service.active_conversations["conversation-1"] = {
            "status": "completed",
            "progress": 100.0,
            "current_step": "Generation completed",
            "lines": [
                {
                    "line_number": 0,
                    "speaker_filename": "speaker.wav",
                    "text": "Hello world",
                    "versions": [
                        {
                            "audio_path": "temp_conversation_segments/line000_v01.wav",
                            "audio_filename": "line000_v01.wav",
                            "similarity_score": 0.90,
                            "robotic_score": 0.10,
                            "quality_score": 0.80,
                            "is_selected": True,
                            "seed": 1234,
                            "seed_origin": "initial",
                            "seed_strategy": "fixed_base_sequential",
                        },
                        {
                            "audio_path": "temp_conversation_segments/line000_v02.wav",
                            "audio_filename": "line000_v02.wav",
                            "similarity_score": 0.95,
                            "robotic_score": 0.05,
                            "quality_score": 0.92,
                            "is_selected": False,
                            "seed": 1235,
                            "seed_origin": "initial",
                            "seed_strategy": "fixed_base_sequential",
                        },
                    ],
                }
            ],
            "parsed_script": [
                {
                    "speaker_filename": "speaker.wav",
                    "text": "Hello world",
                    "line_number": 0,
                    "emotion_vectors": LINE_EMOTION_VECTOR,
                }
            ],
            "generation_params": {
                "versions_per_line": 2,
                "similarity_threshold": 0.6,
                "seed_strategy": "fixed_base_sequential",
                "fixed_base_seed": 1234,
            },
            "seed_runtime_metadata": {
                "seed_strategy": "fixed_base_sequential",
                "fixed_base_seed": 1234,
                "resolved_base_seed": 1234,
                "reused_seed_list": [],
            },
        }

        ui_state = {
            "conversationTitle": "Demo Project",
            "scriptText": "speaker.wav: Hello world",
            "parsedScript": {
                "title": "Demo Project",
                "lines": [
                    {
                        "speaker_filename": "speaker.wav",
                        "text": "Hello world",
                        "line_number": 0,
                        "emo_vector": LINE_EMOTION_VECTOR,
                    }
                ],
            },
            "conversationScript": [
                {
                    "speaker_filename": "speaker.wav",
                    "speaker": "speaker",
                    "text": "Hello world",
                    "line_number": 0,
                    "emo_vector": LINE_EMOTION_VECTOR,
                }
            ],
            "currentConversationId": "conversation-1",
            "currentConversationData": {
                "conversation_id": "conversation-1",
                "status": "completed",
                "lines": [
                    {
                        "line_number": 0,
                        "speaker_filename": "speaker.wav",
                        "text": "Hello world",
                        "versions": [
                            {
                                "audio_path": "temp_conversation_segments/line000_v01.wav",
                                "audio_filename": "line000_v01.wav",
                                "similarity_score": 0.90,
                                "robotic_score": 0.10,
                                "quality_score": 0.80,
                                "is_selected": False,
                                "seed": 1234,
                                "seed_origin": "initial",
                                "seed_strategy": "fixed_base_sequential",
                            },
                            {
                                "audio_path": "temp_conversation_segments/line000_v02.wav",
                                "audio_filename": "line000_v02.wav",
                                "similarity_score": 0.95,
                                "robotic_score": 0.05,
                                "quality_score": 0.92,
                                "is_selected": True,
                                "seed": 1235,
                                "seed_origin": "initial",
                                "seed_strategy": "fixed_base_sequential",
                            },
                        ],
                    }
                ],
            },
            "generationSettings": {
                "versions_per_line": 2,
                "similarity_threshold": 0.6,
                "seed_strategy": "fixed_base_sequential",
                "fixed_base_seed": 1234,
            },
            "currentTab": "conversation-results",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            with patch("backend.api.services.conversation_service.SAVE_DIR", save_dir):
                save_result = service.save_project_state(ui_state, "Demo Project")

                self.assertEqual(save_result["save_name"], "Demo_Project.json")
                self.assertTrue((save_dir / "Demo_Project.json").is_file())

                service.active_conversations = {}
                loaded = service.load_project_state("Demo_Project.json")

        self.assertEqual(loaded["restored_conversation_id"], "conversation-1")
        restored = service.active_conversations["conversation-1"]
        self.assertTrue(restored["lines"][0]["versions"][1]["is_selected"])
        self.assertEqual(restored["parsed_script"][0]["emotion_vectors"], LINE_EMOTION_VECTOR)
        self.assertEqual(restored["generation_params"]["seed_strategy"], "fixed_base_sequential")
        self.assertEqual(restored["generation_params"]["fixed_base_seed"], 1234)
        self.assertEqual(restored["lines"][0]["versions"][0]["seed"], 1234)
        self.assertEqual(
            loaded["project_data"]["ui_state"]["currentConversationId"],
            "conversation-1",
        )

    def test_list_project_saves_returns_project_metadata(self):
        service = ConversationService(conversation_manager=None)
        ui_state = {
            "conversationTitle": "Metadata Demo",
            "scriptText": "speaker.wav: Hi there",
            "parsedScript": {
                "title": "Metadata Demo",
                "lines": [
                    {
                        "speaker_filename": "speaker.wav",
                        "text": "Hi there",
                        "line_number": 0,
                    }
                ],
            },
            "conversationScript": [
                {
                    "speaker_filename": "speaker.wav",
                    "speaker": "speaker",
                    "text": "Hi there",
                    "line_number": 0,
                }
            ],
            "generationSettings": {
                "versions_per_line": 1,
            },
            "currentTab": "conversation-workflow",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            with patch("backend.api.services.conversation_service.SAVE_DIR", save_dir):
                service.save_project_state(ui_state, "metadata-demo")
                projects = service.list_project_saves()

        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0]["save_name"], "metadata-demo.json")
        self.assertEqual(projects[0]["title"], "Metadata Demo")
        self.assertEqual(projects[0]["total_lines"], 1)
        self.assertFalse(projects[0]["has_conversation"])


if __name__ == "__main__":
    unittest.main()
