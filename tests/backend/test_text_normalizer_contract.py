import unittest

from indextts.utils.front import TextNormalizer


class TextNormalizerContractTests(unittest.TestCase):
    def test_text_normalizer_loads_wetext_backend(self):
        normalizer = TextNormalizer()
        normalizer.load()

        self.assertIsNotNone(normalizer.zh_normalizer)
        self.assertIsNotNone(normalizer.en_normalizer)
        self.assertIn(normalizer.backend_name, {"wetext", "wetext-linux-tn"})

    def test_text_normalizer_normalizes_simple_english_numbers(self):
        normalizer = TextNormalizer()
        normalizer.load()

        normalized = normalizer.normalize("I have 2 apples.")

        self.assertEqual(normalized, "I have two apples.")


if __name__ == "__main__":
    unittest.main()
