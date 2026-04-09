import unittest

from backend.api.routers.conversation import (
    summarize_line_selection_state,
    build_selection_gating_message,
)


class SelectionGatingContractTests(unittest.TestCase):
    def test_selection_summary_requires_exactly_one_version_per_line(self):
        lines = [
            {
                "line_number": 0,
                "versions": [
                    {"is_selected": True},
                    {"is_selected": False},
                ],
            },
            {
                "line_number": 1,
                "versions": [
                    {"is_selected": False},
                    {"is_selected": False},
                ],
            },
        ]

        summary = summarize_line_selection_state(lines)

        self.assertFalse(summary["can_export"])
        self.assertEqual(summary["selected_line_count"], 1)
        self.assertEqual(summary["missing_line_numbers"], [2])
        self.assertEqual(summary["multi_selected_line_numbers"], [])

    def test_selection_summary_rejects_multiple_selections_for_single_line(self):
        lines = [
            {
                "line_number": 2,
                "versions": [
                    {"is_selected": True},
                    {"is_selected": True},
                ],
            }
        ]

        summary = summarize_line_selection_state(lines)

        self.assertFalse(summary["can_export"])
        self.assertEqual(summary["missing_line_numbers"], [])
        self.assertEqual(summary["multi_selected_line_numbers"], [3])

    def test_selection_gating_message_lists_missing_and_multi_lines(self):
        message = build_selection_gating_message(
            {
                "missing_line_numbers": [1, 4],
                "multi_selected_line_numbers": [3],
            }
        )

        self.assertIn("lines 1, 4", message)
        self.assertIn("lines 3", message)

    def test_selection_summary_allows_export_when_every_line_has_one_selection(self):
        lines = [
            {"line_number": 0, "versions": [{"is_selected": True}]},
            {"line_number": 1, "versions": [{"is_selected": False}, {"is_selected": True}]},
        ]

        summary = summarize_line_selection_state(lines)

        self.assertTrue(summary["can_export"])
        self.assertEqual(summary["selected_line_count"], 2)
        self.assertEqual(summary["missing_line_numbers"], [])
        self.assertEqual(summary["multi_selected_line_numbers"], [])


if __name__ == "__main__":
    unittest.main()
