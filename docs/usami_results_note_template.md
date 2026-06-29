# Dark Current MLX Pilot - Results Note Template

Hi Professor Usami,

I refactored the MLX prototype into a reproducible local harness and ran a first small pilot.

## Setup

- Repo:
- Commit:
- Machine:
- Judge model:
- Prompt template: `pairwise_judge_v1`
- Generation config:
  - `temperature=0.0`
  - `seed=42`
  - `max_tokens=256`

## Command

```bash
python3 eval_pipeline.py \
  --runner mlx \
  --model MODEL_ID \
  --input DATASET.jsonl \
  --output results/MODEL_ID.jsonl \
  --report results/MODEL_ID_summary.json
```

## Summary

| Metric | Value |
|---|---:|
| Items | |
| Judge calls | |
| Unknown / malformed calls | |
| Swap-consistent items | |
| Swap-inconsistent items | |
| Dark current score | |

## Notes

- Each item is evaluated twice: original A/B order and swapped B/A order.
- A strict invariant judge should flip `A -> B`, flip `B -> A`, and keep `Tie -> Tie`.
- JSONL records include hardware metadata, generation config, prompt template id, raw output, parsed position winner, and parsed original winner.

## Caveats

- This is a pilot harness, not a full reproduction of the paper.
- The dataset slice is intentionally small until we align on the preferred benchmark/schema.
- No proprietary API calls were used.

Best,
Orazio
