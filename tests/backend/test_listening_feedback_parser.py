import unittest

from backend.scripts.parse_listening_feedback import parse_feedback_text, summarize_feedback


class ListeningFeedbackParserTests(unittest.TestCase):
    def test_parser_supports_multiple_blocks_and_aliases(self):
        sample = """
CLIP=b4a5cd17/L1/V2
OVERALL=bad
SIMILARITY=2
NATURALNESS=2
PACE=1
ROBOTIC=5
CLARITY=3
EMOTION=2
ISSUES=too_fast,robotic,weak_similarity
ACTIONS=more_faithful,slower
NOTES=Rushed and metallic.

CLIP=b4a5cd17/L1/V3
VERDICT=ok
SIMILARITY=3
NATURALNESS=3
PACE=3
ROBOTIC=3
CLARITY=4
EMOTION=3
ISSUES=slightly_flat
ACTION=keep_testing
"""

        entries = parse_feedback_text(sample)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["verdict"], "bad")
        self.assertEqual(entries[0]["issues"], ["too_fast", "robotic", "weak_similarity"])
        self.assertEqual(entries[0]["action"], ["more_faithful", "slower"])

        summary = summarize_feedback(entries)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["verdict_counts"]["bad"], 1)
        self.assertEqual(summary["average_scores"]["similarity"], 2.5)
        self.assertTrue(any("Clone Fidelity" in item for item in summary["recommendations"]))

    def test_parser_requires_clip_and_verdict(self):
        with self.assertRaises(ValueError):
            parse_feedback_text("VERDICT=bad\nSIMILARITY=2")

        with self.assertRaises(ValueError):
            parse_feedback_text("CLIP=b4a5cd17/L1/V2\nSIMILARITY=2")


if __name__ == "__main__":
    unittest.main()
