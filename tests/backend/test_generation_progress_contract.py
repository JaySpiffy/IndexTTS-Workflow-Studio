import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.api.core.conversation_manager import ConversationManager
from backend.api.services.conversation_service import ConversationService


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


class ProgressEmittingFakeTTS(FakeTTS):
    def __init__(self):
        super().__init__()
        self.progress_callback = None

    def infer(self, speaker_prompt, text, output_path, **kwargs):
        self.calls.append(
            {
                "speaker_prompt": speaker_prompt,
                "text": text,
                "output_path": output_path,
                **kwargs,
            }
        )
        if callable(self.progress_callback):
            self.progress_callback(0.0, desc="starting inference...")
            time.sleep(0.05)
            self.progress_callback(0.45, desc="planning acoustics 1/1...")
            time.sleep(0.05)
            self.progress_callback(0.88, desc="vocoding segment 1/1...")
            time.sleep(0.05)
        return output_path


class SlowProgressConversationManager:
    def generate_conversation(self, parsed_script, **kwargs):
        yield ("Generating Line 1/2 (speaker.wav)...", "", 12.5, None, None)
        time.sleep(0.2)
        yield ("Attempt 1: Similarity 0.82, Robotic 0.11, Quality 0.76", "", 48.0, None, None)
        time.sleep(0.2)
        yield (
            "Generation completed",
            "",
            100.0,
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


class GenerationProgressContractTests(unittest.TestCase):
    def test_conversation_manager_emits_live_progress_during_inference(self):
        fake_tts = ProgressEmittingFakeTTS()
        manager = ConversationManager(
            tts_core=SimpleNamespace(tts=fake_tts),
            cmd_args=SimpleNamespace(verbose=False),
        )
        live_progress_updates = []

        with patch("backend.api.core.file_utils.prepare_temp_dir", return_value=True), patch(
            "backend.api.core.audio_processing.analyze_speaker_similarity_with_quality",
            return_value={"similarity": 0.96, "robotic_score": 0.05, "quality_score": 0.91},
        ):
            list(
                manager.generate_conversation(
                    parsed_script=[
                        {
                            "speaker_filename": "speaker.wav",
                            "text": "Live progress contract line.",
                            "line_number": 0,
                        }
                    ],
                    versions_per_line=1,
                    similarity_threshold=0.90,
                    robotic_threshold=0.10,
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
                    progress=lambda value, step: live_progress_updates.append((value, step)),
                )
            )

        self.assertGreaterEqual(len(live_progress_updates), 3)
        self.assertTrue(any("planning acoustics" in step for _, step in live_progress_updates))
        self.assertTrue(any(value > 40.0 for value, _ in live_progress_updates))

    def test_conversation_manager_reports_incremental_progress_before_completion(self):
        fake_tts = FakeTTS()
        manager = ConversationManager(
            tts_core=SimpleNamespace(tts=fake_tts),
            cmd_args=SimpleNamespace(verbose=False),
        )

        with patch("backend.api.core.file_utils.prepare_temp_dir", return_value=True), patch(
            "backend.api.core.audio_processing.analyze_speaker_similarity_with_quality",
            return_value={"similarity": 0.42, "robotic_score": 0.82, "quality_score": 0.38},
        ):
            updates = list(
                manager.generate_conversation(
                    parsed_script=[
                        {
                            "speaker_filename": "speaker.wav",
                            "text": "Progress contract test line.",
                            "line_number": 0,
                        }
                    ],
                    versions_per_line=2,
                    similarity_threshold=0.95,
                    robotic_threshold=0.10,
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
                )
            )

        progress_values = [progress for _, _, progress, _, _ in updates]
        self.assertGreater(progress_values[2], 0.0)
        self.assertGreater(progress_values[-2], progress_values[2])
        self.assertEqual(progress_values[-1], 100.0)

    def test_generate_conversation_async_keeps_status_pollable_while_running(self):
        service = ConversationService(conversation_manager=SlowProgressConversationManager())
        parsed_script = [
            {
                "speaker_filename": "speaker.wav",
                "text": "Async progress contract test.",
                "line_number": 0,
            }
        ]

        async def run_contract():
            with patch("backend.api.core.file_utils.validate_speaker_files", return_value=[]):
                task_info = service.start_conversation_generation(parsed_script=parsed_script)

            background_task = asyncio.create_task(
                service.generate_conversation_async(task_info["conversation_id"])
            )

            await asyncio.sleep(0.05)
            live_status = service.get_conversation_status(task_info["conversation_id"])

            self.assertFalse(background_task.done())
            self.assertEqual(live_status["status"], "running")
            self.assertGreater(live_status["progress_percent"], 0.0)
            self.assertIn("Generating Line", live_status["current_step"])

            result = await background_task
            self.assertEqual(result["status"], "completed")

        asyncio.run(run_contract())

    def test_generate_conversation_async_surfaces_live_inference_progress(self):
        service = ConversationService(
            conversation_manager=ConversationManager(
                tts_core=SimpleNamespace(tts=ProgressEmittingFakeTTS()),
                cmd_args=SimpleNamespace(verbose=False),
            )
        )
        parsed_script = [
            {
                "speaker_filename": "speaker.wav",
                "text": "Async live progress contract test.",
                "line_number": 0,
            }
        ]

        async def run_contract():
            with patch("backend.api.core.file_utils.validate_speaker_files", return_value=[]), patch(
                "backend.api.core.file_utils.prepare_temp_dir",
                return_value=True,
            ), patch(
                "backend.api.core.audio_processing.analyze_speaker_similarity_with_quality",
                return_value={"similarity": 0.97, "robotic_score": 0.04, "quality_score": 0.93},
            ):
                task_info = service.start_conversation_generation(
                    parsed_script=parsed_script,
                    versions_per_line=1,
                    auto_regen_attempts=0,
                )
                background_task = asyncio.create_task(
                    service.generate_conversation_async(task_info["conversation_id"])
                )

                await asyncio.sleep(0.08)
                live_status = service.get_conversation_status(task_info["conversation_id"])

                self.assertFalse(background_task.done())
                self.assertEqual(live_status["status"], "running")
                self.assertGreater(live_status["progress_percent"], 30.0)
                self.assertIn("planning acoustics", live_status["current_step"])

                result = await background_task
                self.assertEqual(result["status"], "completed")

        asyncio.run(run_contract())


if __name__ == "__main__":
    unittest.main()
