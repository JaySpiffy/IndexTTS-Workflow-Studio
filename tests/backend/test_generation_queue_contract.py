import asyncio
import time
import unittest
from unittest.mock import patch

from backend.api.exceptions import ConversationError
from backend.api.services.conversation_service import ConversationService


class SlowQueueConversationManager:
    def __init__(self, delay_seconds: float = 0.2):
        self.delay_seconds = delay_seconds

    def generate_conversation(self, parsed_script, **kwargs):
        speaker_filename = parsed_script[0]["speaker_filename"]
        text = parsed_script[0]["text"]
        yield (f"Generating Line 1/1 ({speaker_filename})...", "", 10.0, None, None)
        time.sleep(self.delay_seconds)
        yield (
            "Generation completed",
            "",
            100.0,
            [
                {
                    "line_index": 0,
                    "speaker_filename": speaker_filename,
                    "text": text,
                    "versions": [
                        {
                            "audio_path": f"temp_conversation_segments/{speaker_filename}-line.wav",
                            "audio_filename": f"{speaker_filename}-line.wav",
                            "similarity_score": 0.92,
                            "robotic_score": 0.06,
                            "quality_score": 0.88,
                        }
                    ],
                }
            ],
            None,
        )


class GenerationQueueContractTests(unittest.TestCase):
    def test_second_conversation_waits_in_queue_until_slot_is_free(self):
        service = ConversationService(conversation_manager=SlowQueueConversationManager())

        async def run_contract():
            with patch("backend.api.core.file_utils.validate_speaker_files", return_value=[]), patch(
                "backend.api.services.conversation_service.settings.generation_worker_slots",
                1,
            ), patch(
                "backend.api.services.conversation_service.settings.generation_max_pending_tasks",
                2,
            ):
                first = service.start_conversation_generation(
                    parsed_script=[{"speaker_filename": "speaker-a.wav", "text": "First queued job.", "line_number": 0}]
                )
                second = service.start_conversation_generation(
                    parsed_script=[{"speaker_filename": "speaker-b.wav", "text": "Second queued job.", "line_number": 0}]
                )

                first_task = asyncio.create_task(service.generate_conversation_async(first["conversation_id"]))
                await asyncio.sleep(0.03)
                second_task = asyncio.create_task(service.generate_conversation_async(second["conversation_id"]))
                await asyncio.sleep(0.05)

                first_status = service.get_conversation_status(first["conversation_id"])
                second_status = service.get_conversation_status(second["conversation_id"])

                self.assertEqual(first_status["status"], "running")
                self.assertEqual(second_status["status"], "queued")
                self.assertEqual(second_status["queue_position"], 1)
                self.assertIn("Queued for generation", second_status["current_step"])

                await asyncio.gather(first_task, second_task)

                second_finished = service.get_conversation_status(second["conversation_id"])
                self.assertEqual(second_finished["status"], "completed")
                self.assertIsNone(second_finished["queue_position"])

        asyncio.run(run_contract())

    def test_start_generation_rejects_when_pending_queue_is_full(self):
        service = ConversationService(conversation_manager=SlowQueueConversationManager(delay_seconds=0.3))

        async def run_contract():
            with patch("backend.api.core.file_utils.validate_speaker_files", return_value=[]), patch(
                "backend.api.services.conversation_service.settings.generation_worker_slots",
                1,
            ), patch(
                "backend.api.services.conversation_service.settings.generation_max_pending_tasks",
                1,
            ):
                first = service.start_conversation_generation(
                    parsed_script=[{"speaker_filename": "speaker-a.wav", "text": "First queued job.", "line_number": 0}]
                )
                second = service.start_conversation_generation(
                    parsed_script=[{"speaker_filename": "speaker-b.wav", "text": "Second queued job.", "line_number": 0}]
                )

                first_task = asyncio.create_task(service.generate_conversation_async(first["conversation_id"]))
                await asyncio.sleep(0.03)
                second_task = asyncio.create_task(service.generate_conversation_async(second["conversation_id"]))
                await asyncio.sleep(0.05)

                with self.assertRaises(ConversationError):
                    service.start_conversation_generation(
                        parsed_script=[{"speaker_filename": "speaker-c.wav", "text": "Third queued job.", "line_number": 0}]
                    )

                await asyncio.gather(first_task, second_task)

        asyncio.run(run_contract())


if __name__ == "__main__":
    unittest.main()
