"""Per-condition breakdown from run JSONL files.

Each item is order-invariant, order-dependent (both verdicts parsed, and the verdict changes
with the A/B order), or unparseable (at least one call without the strict marker). The strict
`dark_current_score` counts unparseable items as not order-invariant; the split keeps them apart.

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

    # category -> [order-invariant, order-dependent, unparseable]
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for entry in by_item.values():
        verdicts = entry["verdicts"]
        if "ab" not in verdicts or "ba" not in verdicts:
            continue
        if Verdict.UNKNOWN in (verdicts["ab"], verdicts["ba"]):
            state = 2  # a missing marker is a parse failure, not a position flip
        elif is_swap_consistent(verdicts["ab"], verdicts["ba"]):
            state = 0
        else:
            state = 1
        stats[entry["category"]][state] += 1

    print(f"\n{Path(path).name} — judge: {judge}")
    totals = [0, 0, 0]
    for category, counts in sorted(stats.items()):
        invariant, dependent, unparseable = counts
        totals = [total + count for total, count in zip(totals, counts, strict=True)]
        print(
            f"  {category:11}: {invariant}/{sum(counts)} order-invariant, "
            f"{dependent} order-dependent, {unparseable} unparseable"
        )
    invariant, dependent, unparseable = totals
    items = sum(totals)
    print(
        f"  {'TOTAL':11}: {invariant}/{items} order-invariant, "
        f"{dependent} order-dependent ({dependent / items:.3f}), {unparseable} unparseable "
        f"(strict dark_current_score {1 - invariant / items:.3f})"
    )


if __name__ == "__main__":
    for run_path in sys.argv[1:]:
        breakdown(run_path)
