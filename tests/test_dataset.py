from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dark_current_mlx.dataset import load_jsonl


class DatasetTest(unittest.TestCase):
    def test_loads_current_jsonl_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "items.jsonl"
            path.write_text(
                '{"item_id":"x","instruction":"Q","response_a":"A","response_b":"B"}\n',
                encoding="utf-8",
            )
            items = load_jsonl(path)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].item_id, "x")
        self.assertEqual(items[0].response_a, "A")

    def test_loads_legacy_prototype_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "items.jsonl"
            path.write_text(
                '{"id":"x","instruction":"Q","model_a_response":"A","model_b_response":"B"}\n',
                encoding="utf-8",
            )
            items = load_jsonl(path)
        self.assertEqual(items[0].item_id, "x")
        self.assertEqual(items[0].response_b, "B")


if __name__ == "__main__":
    unittest.main()
