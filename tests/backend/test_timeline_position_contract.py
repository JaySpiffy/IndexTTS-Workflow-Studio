import tempfile
import unittest
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from backend.api.core.audio_mixing import mix_audio_files_at_positions
from backend.api.services.timeline_service import TimelineService


class FakeConversationService:
    def __init__(self, lines):
        self.lines = lines

    def get_conversation_status(self, conversation_id):
        return {
            "status": "completed",
            "lines": self.lines,
        }


class TimelinePositionContractTests(unittest.TestCase):
    def test_mix_audio_files_at_positions_respects_explicit_start_times(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first = temp_path / "first.wav"
            second = temp_path / "second.wav"
            output = temp_path / "timeline.wav"

            first_export = AudioSegment.silent(duration=1000).export(first, format="wav")
            second_export = AudioSegment.silent(duration=1000).export(second, format="wav")
            first_export.close()
            second_export.close()

            result = mix_audio_files_at_positions(
                [
                    {"audio_path": str(first), "start_ms": 0},
                    {"audio_path": str(second), "start_ms": 700},
                ],
                str(output),
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["duration_ms"], 1700)
            self.assertEqual(result["timeline"][1]["start_ms"], 700)

    def test_imported_timeline_uses_global_sequence_not_per_speaker_reset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first = temp_path / "first.wav"
            second = temp_path / "second.wav"

            first_export = AudioSegment.silent(duration=1000).export(first, format="wav")
            second_export = AudioSegment.silent(duration=1500).export(second, format="wav")
            first_export.close()
            second_export.close()

            lines = [
                {
                    "speaker_filename": "speaker_a.wav",
                    "text": "Line A1",
                    "versions": [
                        {
                            "audio_path": str(first),
                            "quality_score": 0.8,
                        }
                    ],
                    "best_version_index": 0,
                },
                {
                    "speaker_filename": "speaker_b.wav",
                    "text": "Line B1",
                    "versions": [
                        {
                            "audio_path": str(second),
                            "quality_score": 0.7,
                            "is_selected": True,
                        }
                    ],
                    "best_version_index": 0,
                },
                {
                    "speaker_filename": "speaker_a.wav",
                    "text": "Line A2",
                    "versions": [
                        {
                            "audio_path": str(first),
                            "quality_score": 0.9,
                            "is_selected": True,
                        }
                    ],
                    "best_version_index": 0,
                },
            ]

            service = TimelineService(conversation_service=FakeConversationService(lines))
            service.timeline_dir = temp_path / "timeline_projects"
            service.timeline_dir.mkdir(parents=True, exist_ok=True)

            created = service.create_timeline_project("Import Test", conversation_id="conversation-1")
            project = service.get_timeline_project(created["project_id"])["project"]

            speaker_a_track = next(track for track in project["tracks"] if track["speaker_filename"] == "speaker_a.wav")
            speaker_b_track = next(track for track in project["tracks"] if track["speaker_filename"] == "speaker_b.wav")

            self.assertAlmostEqual(speaker_a_track["segments"][0]["start_time"], 0.0, places=2)
            self.assertAlmostEqual(speaker_b_track["segments"][0]["start_time"], 1.0, places=2)
            self.assertAlmostEqual(speaker_a_track["segments"][1]["start_time"], 2.5, places=2)
            self.assertEqual(speaker_b_track["segments"][0]["audio_filename"], second.name)
            self.assertAlmostEqual(project["total_duration"], 3.5, places=2)

    def test_mix_audio_files_at_positions_applies_track_volume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tone = temp_path / "tone.wav"
            full_output = temp_path / "full.wav"
            reduced_output = temp_path / "reduced.wav"

            tone_segment = Sine(440).to_audio_segment(duration=600).apply_gain(-6)
            tone_export = tone_segment.export(tone, format="wav")
            tone_export.close()

            mix_audio_files_at_positions(
                [{"audio_path": str(tone), "start_ms": 0, "volume": 1.0}],
                str(full_output),
            )
            mix_audio_files_at_positions(
                [{"audio_path": str(tone), "start_ms": 0, "volume": 0.5}],
                str(reduced_output),
            )

            full_mix = AudioSegment.from_file(full_output, format="wav")
            reduced_mix = AudioSegment.from_file(reduced_output, format="wav")

            self.assertLess(reduced_mix.dBFS, full_mix.dBFS - 5.5)

    def test_update_track_volume_persists_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            service = TimelineService()
            service.timeline_dir = temp_path / "timeline_projects"
            service.timeline_dir.mkdir(parents=True, exist_ok=True)

            project_id = "project-volume"
            service.active_projects[project_id] = {
                "project": {
                    "project_id": project_id,
                    "project_name": "Volume Test",
                    "description": None,
                    "conversation_id": None,
                    "tracks": [
                        {
                            "track_id": "track-1",
                            "track_name": "Lead",
                            "speaker_filename": "Pr.D.Trump.wav",
                            "segments": [],
                            "volume": 1.0,
                            "muted": False,
                            "solo": False,
                        }
                    ],
                    "total_duration": 0.0,
                    "created_at": "2026-04-03 00:00:00",
                    "updated_at": "2026-04-03 00:00:00",
                },
                "status": "created",
                "created_at": 0.0,
                "updated_at": 0.0,
            }

            result = service.update_track_volume(project_id, "track-1", 0.65)

            self.assertAlmostEqual(result["volume"], 0.65, places=3)
            project = service.get_timeline_project(project_id)["project"]
            self.assertAlmostEqual(project["tracks"][0]["volume"], 0.65, places=3)
            self.assertTrue((service.timeline_dir / f"{project_id}.json").exists())


if __name__ == "__main__":
    unittest.main()
