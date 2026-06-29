from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dark_current_mlx.dataset import iter_builtin_sample
from dark_current_mlx.models import GenerationConfig
from dark_current_mlx.pipeline import run_items
from dark_current_mlx.runners import MockRunner


class PipelineTest(unittest.TestCase):
    def test_mock_position_runner_produces_inconsistency_report(self):
        records, summary = run_items(
            items=list(iter_builtin_sample()),
            runner=MockRunner("position"),
            judge_model="mock:position",
            generation_config=GenerationConfig(),
        )

        self.assertEqual(len(records), 4)
        self.assertEqual(summary["total_items"], 2)
        self.assertEqual(summary["swap_inconsistent_items"], 2)
        self.assertEqual(summary["dark_current_score"], 1.0)

    def test_mock_tie_runner_is_consistent(self):
        records, summary = run_items(
            items=list(iter_builtin_sample()),
            runner=MockRunner("tie"),
            judge_model="mock:tie",
            generation_config=GenerationConfig(),
        )

        self.assertEqual(len(records), 4)
        self.assertEqual(summary["swap_consistent_items"], 2)
        self.assertEqual(summary["dark_current_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
