"""Per-condition dark current breakdown from run JSONL files.

Usage: python3 scripts/category_breakdown.py results/<run>.jsonl [...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dark_current_mlx.models import Verdict  # noqa: E402
from dark_current_mlx.verdicts import is_swap_consistent  # noqa: E402


def breakdown(path: str) -> None:
    by_item: dict[str, dict[str, object]] = {}
    judge = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            judge = record["judge_model"]
            entry = by_item.setdefault(
                record["item_id"],
                {"category": record["inputs"]["metadata"].get("category", "?"), "verdicts": {}},
            )
            entry["verdicts"][record["order"]] = Verdict(
                record["outputs"]["parsed_position_winner"]
            )

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # category -> [consistent, total]
    for entry in by_item.values():
        verdicts = entry["verdicts"]
        if "ab" not in verdicts or "ba" not in verdicts:
            continue
        consistent = is_swap_consistent(verdicts["ab"], verdicts["ba"])
        stats[entry["category"]][0] += int(consistent)
        stats[entry["category"]][1] += 1

    print(f"\n{Path(path).name} — judge: {judge}")
    total_consistent = total_items = 0
    for category, (consistent, total) in sorted(stats.items()):
        total_consistent += consistent
        total_items += total
        print(f"  {category:11}: {consistent}/{total} consistent (dark current {1 - consistent / total:.2f})")
    print(f"  {'TOTAL':11}: {total_consistent}/{total_items} (dark_current_score {1 - total_consistent / total_items:.3f})")


if __name__ == "__main__":
    for run_path in sys.argv[1:]:
        breakdown(run_path)
