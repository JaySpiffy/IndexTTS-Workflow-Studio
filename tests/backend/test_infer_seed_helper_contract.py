import random
import unittest

import numpy as np
import torch

from backend.indextts.infer_v2 import IndexTTS2


class InferSeedHelperContractTests(unittest.TestCase):
    def setUp(self):
        self.tts = IndexTTS2.__new__(IndexTTS2)

    def assertNumpyStateEqual(self, left, right):
        self.assertEqual(left[0], right[0])
        self.assertTrue(np.array_equal(left[1], right[1]))
        self.assertEqual(left[2], right[2])
        self.assertEqual(left[3], right[3])
        self.assertEqual(left[4], right[4])

    def test_seed_helper_repeats_draws_and_restores_rng_state(self):
        python_state_before = random.getstate()
        numpy_state_before = np.random.get_state()
        torch_state_before = torch.random.get_rng_state()

        def draw_values():
            return (
                random.randint(0, 10_000),
                int(np.random.randint(0, 10_000)),
                int(torch.randint(0, 10_000, (1,)).item()),
            )

        first = self.tts._run_with_deterministic_seed(1234, draw_values)

        python_state_after_first = random.getstate()
        numpy_state_after_first = np.random.get_state()
        torch_state_after_first = torch.random.get_rng_state()

        second = self.tts._run_with_deterministic_seed(1234, draw_values)

        self.assertEqual(first, second)
        self.assertEqual(python_state_before, python_state_after_first)
        self.assertNumpyStateEqual(numpy_state_before, numpy_state_after_first)
        self.assertTrue(torch.equal(torch_state_before, torch_state_after_first))


if __name__ == "__main__":
    unittest.main()
