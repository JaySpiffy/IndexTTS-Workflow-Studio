import tempfile
import unittest
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from backend.api.core.audio_mixing import mix_audio_files_at_positions
from backend.api.services.timeline_service import TimelineService


class TimelineEditorContractTests(unittest.TestCase):
    def test_mix_audio_files_at_positions_ducks_later_overlap_segment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tone_path = temp_path / "tone.wav"
            plain_output = temp_path / "plain.wav"
            ducked_output = temp_path / "ducked.wav"

            source_tone = Sine(440).to_audio_segment(duration=1000).apply_gain(-9)
            exported = source_tone.export(tone_path, format="wav")
            exported.close()

            placements = [
                {"audio_path": str(tone_path), "start_ms": 0, "track_id": "lead", "segment_id": "a"},
                {"audio_path": str(tone_path), "start_ms": 500, "track_id": "interrupt", "segment_id": "b"},
            ]

            mix_audio_files_at_positions(placements, str(plain_output))
            mix_audio_files_at_positions(
                placements,
                str(ducked_output),
                duck_overlaps=True,
                duck_amount_db=6.0,
                duck_fade_ms=0,
            )

            with open(plain_output, "rb") as plain_handle:
                plain_mix = AudioSegment.from_file(plain_handle, format="wav")
            with open(ducked_output, "rb") as ducked_handle:
                ducked_mix = AudioSegment.from_file(ducked_handle, format="wav")

            plain_overlap = plain_mix[550:900]
            ducked_overlap = ducked_mix[550:900]
            self.assertLess(ducked_overlap.dBFS, plain_overlap.dBFS - 2.0)

    def test_split_segment_creates_two_regen_required_segments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            service = TimelineService()
            service.timeline_dir = temp_path / "timeline_projects"
            service.timeline_dir.mkdir(parents=True, exist_ok=True)

            project_id = "timeline-split"
            service.active_projects[project_id] = {
                "project": {
                    "project_id": project_id,
                    "project_name": "Split Test",
                    "description": None,
                    "conversation_id": None,
                    "tracks": [
                        {
                            "track_id": "track-1",
                            "track_name": "Lead",
                            "speaker_filename": "Pr.D.Trump.wav",
                            "segments": [
                                {
                                    "segment_id": "segment-1",
                                    "text": "These are not flowers they are human lives",
                                    "speaker_filename": "Pr.D.Trump.wav",
                                    "start_time": 0.0,
                                    "duration": 4.0,
                                    "audio_filename": "segment.wav",
                                    "emotion_control_method": "from_speaker",
                                    "emotion_weight": 1.0,
                                    "emotion_vectors": [],
                                    "emotion_text": None,
                                    "use_random_sampling": False,
                                    "emotion_keyframes": [],
                                    "emotion_interpolation_type": "linear",
                                    "emotion_transition_duration": 0.5,
                                    "emotion_timeline_enabled": False,
                                    "max_text_tokens_per_segment": 120,
                                    "do_sample": True,
                                    "top_p": 0.8,
                                    "top_k": 30,
                                    "temperature": 0.8,
                                    "length_penalty": 0.0,
                                    "num_beams": 3,
                                    "repetition_penalty": 10.0,
                                    "max_mel_tokens": 1500,
                                }
                            ],
                            "volume": 1.0,
                            "muted": False,
                            "solo": False,
                        }
                    ],
                    "total_duration": 4.0,
                    "created_at": "2026-04-03 00:00:00",
                    "updated_at": "2026-04-03 00:00:00",
                },
                "status": "created",
                "created_at": 0.0,
                "updated_at": 0.0,
            }

            result = service.split_segment(project_id, "track-1", "segment-1", split_offset=1.5)
            project = service.get_timeline_project(project_id)["project"]
            segments = project["tracks"][0]["segments"]

            self.assertEqual(len(segments), 2)
            self.assertEqual(result["updated_segment"]["audio_filename"], None)
            self.assertEqual(result["new_segment"]["audio_filename"], None)
            self.assertAlmostEqual(segments[0]["duration"], 1.5, places=2)
            self.assertAlmostEqual(segments[1]["start_time"], 1.5, places=2)
            self.assertAlmostEqual(segments[1]["duration"], 2.5, places=2)
            self.assertTrue((service.timeline_dir / f"{project_id}.json").exists())

    def test_get_segment_waveform_returns_requested_bar_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tone_path = temp_path / "waveform-tone.wav"
            exported = Sine(220).to_audio_segment(duration=900).apply_gain(-12).export(tone_path, format="wav")
            exported.close()

            service = TimelineService()
            service.timeline_dir = temp_path / "timeline_projects"
            service.timeline_dir.mkdir(parents=True, exist_ok=True)

            project_id = "timeline-waveform"
            service.active_projects[project_id] = {
                "project": {
                    "project_id": project_id,
                    "project_name": "Waveform Test",
                    "description": None,
                    "conversation_id": None,
                    "tracks": [
                        {
                            "track_id": "track-1",
                            "track_name": "Lead",
                            "speaker_filename": "JoeRogan.wav",
                            "segments": [
                                {
                                    "segment_id": "segment-1",
                                    "text": "Waveform me please",
                                    "speaker_filename": "JoeRogan.wav",
                                    "start_time": 0.0,
                                    "duration": 0.9,
                                    "audio_filename": str(tone_path),
                                }
                            ],
                            "volume": 1.0,
                            "muted": False,
                            "solo": False,
                        }
                    ],
                    "total_duration": 0.9,
                    "created_at": "2026-04-03 00:00:00",
                    "updated_at": "2026-04-03 00:00:00",
                },
                "status": "created",
                "created_at": 0.0,
                "updated_at": 0.0,
            }

            waveform = service.get_segment_waveform(project_id, "track-1", "segment-1", bars=32)

            self.assertEqual(waveform["bar_count"], 32)
            self.assertEqual(len(waveform["peaks"]), 32)
            self.assertTrue(any(peak > 0.05 for peak in waveform["peaks"]))


if __name__ == "__main__":
    unittest.main()
