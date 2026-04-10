import unittest
from unittest.mock import patch

from backend.api.core import audio_processing


class SimilarityBackendContractTests(unittest.TestCase):
    def test_fusion_backend_averages_speechbrain_and_campplus_scores(self):
        with patch.object(audio_processing, "analyze_speaker_similarity", return_value=0.82), patch.object(
            audio_processing, "initialize_campplus_similarity_model", return_value=True
        ), patch.object(
            audio_processing, "analyze_speaker_similarity_campplus", return_value=0.62
        ), patch.object(
            audio_processing, "detect_robotic_speech", return_value=0.2
        ):
            result = audio_processing.analyze_speaker_similarity_with_quality(
                object(),
                "ref.wav",
                "gen.wav",
                similarity_backend="fusion",
            )

        self.assertAlmostEqual(result["similarity"], 0.72, places=2)
        self.assertAlmostEqual(result["quality_score"], 0.648, places=3)
        self.assertEqual(result["similarity_backend_used"], "fusion")
        self.assertEqual(result["similarity_requested_backend"], "fusion")
        self.assertAlmostEqual(result["similarity_breakdown"]["speechbrain"], 0.82, places=2)
        self.assertAlmostEqual(result["similarity_breakdown"]["campplus"], 0.62, places=2)

    def test_auto_backend_falls_back_to_campplus_when_speechbrain_is_unavailable(self):
        with patch.object(audio_processing, "analyze_speaker_similarity", return_value=-1.0), patch.object(
            audio_processing, "initialize_campplus_similarity_model", return_value=True
        ), patch.object(
            audio_processing, "analyze_speaker_similarity_campplus", return_value=0.77
        ), patch.object(
            audio_processing, "detect_robotic_speech", return_value=0.1
        ):
            result = audio_processing.analyze_speaker_similarity_with_quality(
                None,
                "ref.wav",
                "gen.wav",
                similarity_backend="auto",
            )

        self.assertAlmostEqual(result["similarity"], 0.77, places=2)
        self.assertAlmostEqual(result["quality_score"], 0.7315, places=4)
        self.assertEqual(result["similarity_backend_used"], "campplus")
        self.assertEqual(result["similarity_requested_backend"], "auto")
        self.assertIsNone(result["similarity_breakdown"]["speechbrain"])
        self.assertAlmostEqual(result["similarity_breakdown"]["campplus"], 0.77, places=2)


if __name__ == "__main__":
    unittest.main()
