import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydub.generators import Sine

from backend.api.core.source_clip_prep import prepare_source_clip


def build_noisy_clip():
    base = Sine(440).to_audio_segment(duration=3000).apply_gain(-14)
    return base.set_channels(1)


class DeepFilterIntegrationContractTests(unittest.TestCase):
    def test_prepare_source_clip_uses_deepfilter_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "noisy.wav"
            output_path = temp_path / "clean.wav"
            export_handle = build_noisy_clip().export(source_path, format="wav")
            export_handle.close()

            def fake_deepfilter(audio, sample_rate, noise_reduction_strength=0.35):
                self.assertEqual(sample_rate, 44100)
                self.assertAlmostEqual(noise_reduction_strength, 0.55, places=2)
                return np.asarray(audio, dtype=np.float32) * 0.9

            with patch("backend.api.core.source_clip_prep.DEEPFILTERNET_AVAILABLE", True), patch(
                "backend.api.core.source_clip_prep.apply_deepfilter_noise_reduction",
                side_effect=fake_deepfilter,
            ):
                result = prepare_source_clip(
                    source_path,
                    output_path,
                    convert_to_mono=True,
                    normalize_audio=False,
                    use_noise_reduction=True,
                    noise_reduction_strength=0.55,
                    noise_reduction_backend="deepfilter",
                )

            self.assertTrue(output_path.exists())
            self.assertTrue(
                any("DeepFilterNet speech cleanup" in note for note in result["processing_notes"])
            )
            self.assertEqual(
                result["applied_options"]["noise_reduction_backend"],
                "deepfilter",
            )

    def test_prepare_source_clip_falls_back_to_classic_noise_reduction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "noisy.wav"
            output_path = temp_path / "clean.wav"
            export_handle = build_noisy_clip().export(source_path, format="wav")
            export_handle.close()

            with patch("backend.api.core.source_clip_prep.DEEPFILTERNET_AVAILABLE", True), patch(
                "backend.api.core.source_clip_prep.apply_deepfilter_noise_reduction",
                return_value=None,
            ), patch("backend.api.core.source_clip_prep.NOISEREDUCE_AVAILABLE", True), patch(
                "backend.api.core.source_clip_prep.nr.reduce_noise",
                side_effect=lambda y, sr, prop_decrease: np.asarray(y, dtype=np.float32),
            ):
                result = prepare_source_clip(
                    source_path,
                    output_path,
                    convert_to_mono=True,
                    normalize_audio=False,
                    use_noise_reduction=True,
                    noise_reduction_strength=0.4,
                    noise_reduction_backend="auto",
                )

            self.assertTrue(output_path.exists())
            self.assertTrue(
                any("classic noise reduction was used instead" in note for note in result["processing_notes"])
            )


if __name__ == "__main__":
    unittest.main()
