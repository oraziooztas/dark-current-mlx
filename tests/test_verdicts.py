from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dark_current_mlx.models import Verdict
from dark_current_mlx.verdicts import extract_verdict, is_swap_consistent, original_winner


class VerdictParsingTest(unittest.TestCase):
    def test_extracts_last_strict_marker(self):
        text = "Reasoning mentions [[A]] first, final answer is [[B]]."
        self.assertEqual(extract_verdict(text), Verdict.B)

    def test_unknown_when_marker_missing(self):
        self.assertEqual(extract_verdict("A is probably better."), Verdict.UNKNOWN)

    def test_swap_consistency(self):
        self.assertTrue(is_swap_consistent(Verdict.A, Verdict.B))
        self.assertTrue(is_swap_consistent(Verdict.B, Verdict.A))
        self.assertTrue(is_swap_consistent(Verdict.TIE, Verdict.TIE))
        self.assertFalse(is_swap_consistent(Verdict.A, Verdict.A))
        self.assertFalse(is_swap_consistent(Verdict.UNKNOWN, Verdict.A))

    def test_original_winner_mapping(self):
        self.assertEqual(original_winner(Verdict.A, "ab"), "model_a")
        self.assertEqual(original_winner(Verdict.B, "ab"), "model_b")
        self.assertEqual(original_winner(Verdict.A, "ba"), "model_b")
        self.assertEqual(original_winner(Verdict.B, "ba"), "model_a")


if __name__ == "__main__":
    unittest.main()
