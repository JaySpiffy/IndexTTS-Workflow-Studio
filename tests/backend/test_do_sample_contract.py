import unittest

import torch

from backend.indextts.infer_v2 import IndexTTS2


class StopAfterInference(Exception):
    pass


class FakeTokenizer:
    def tokenize(self, text):
        return ["hello"]

    def split_segments(self, text_tokens_list, max_text_tokens_per_segment):
        return [["hello"]]

    def convert_tokens_to_ids(self, sent):
        return [1, 2, 3]


class RecordingGPT:
    def __init__(self):
        self.calls = []

    def merge_emovec(self, *args, **kwargs):
        return torch.zeros(1, 4)

    def inference_speech(self, *args, **kwargs):
        self.calls.append(kwargs)
        raise StopAfterInference()


class InferV2SamplingContractTests(unittest.TestCase):
    def test_infer_honors_do_sample_flag(self):
        fake_tts = type("FakeTTS", (), {})()
        fake_tts._set_progress = lambda *args, **kwargs: None
        fake_tts.qwen_emo = None
        fake_tts.cache_spk_cond = torch.zeros(1, 1, 4)
        fake_tts.cache_spk_audio_prompt = "prompt.wav"
        fake_tts.cache_s2mel_style = torch.zeros(1, 192)
        fake_tts.cache_s2mel_prompt = torch.zeros(1, 1, 4)
        fake_tts.cache_mel = torch.zeros(1, 80, 10)
        fake_tts.cache_emo_cond = torch.zeros(1, 1, 4)
        fake_tts.cache_emo_audio_prompt = "prompt.wav"
        fake_tts.tokenizer = FakeTokenizer()
        fake_tts.device = "cpu"
        fake_tts.dtype = None
        fake_tts.gpt = RecordingGPT()

        with self.assertRaises(StopAfterInference):
            IndexTTS2.infer(
                fake_tts,
                spk_audio_prompt="prompt.wav",
                text="Hello there.",
                output_path="ignored.wav",
                do_sample=False,
                verbose=False,
            )

        self.assertEqual(len(fake_tts.gpt.calls), 1)
        self.assertIs(fake_tts.gpt.calls[0]["do_sample"], False)


if __name__ == "__main__":
    unittest.main()
