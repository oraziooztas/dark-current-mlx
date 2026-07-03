# Dark Current MLX Pilot - Results Note (draft 2026-07-03, not sent)

Hi Professor Usami,

I ran the first real mini-slice on the local MLX harness while waiting on your preferred benchmark slice. Three open-weight judges, fully local, deterministic config.

## Setup

- Repo: https://github.com/oraziooztas/dark-current-mlx
- Commit: FILL_AT_SEND_TIME
- Machine: MacBook Pro (Apple M3 Pro, 18 GB), Python 3.12.13, mlx-lm 0.31.3
- Judges: mlx-community 4-bit builds of Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2-7B-Instruct
- Prompt template: `pairwise_judge_v1`
- Generation config: `temperature=0.0`, `seed=42`, `max_tokens=512`
- Dataset: `data/minislice_v0.jsonl`, 12 hand-written pairwise items in three conditions: `clear_gap` (one answer plainly better, better side balanced 2/2), `identical` (byte-identical responses), `paraphrase` (same content, different wording). 24 judge calls per model, both presentation orders.

## Summary

| Judge (4-bit) | clear_gap | identical | paraphrase | overall | unknown calls | gen tokens | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3-8B-Instruct | 0.25* | 0.25 | 0.50 | 0.333 | 1 | 4,459 | 170 s |
| Mistral-7B-v0.3 | 0.00 | 0.00 | 1.00 | 0.333 | 1 | 2,725 | 106 s |
| Qwen2-7B | 0.00 | 0.50 | 0.75 | 0.417 | 0 | 1,508 | 71 s |

Cells are per-condition dark current, the fraction of items whose verdict pair is not order-invariant under the A/B swap. *One Llama call gave the right verdict in prose but skipped the strict [[B]] marker, so it parses as unknown.

## What stood out

1. All three judges detect a real quality gap well: parsed clear_gap verdicts picked the objectively better answer in both orders 11/12 times.
2. Position bias shows up exactly where the quality signal disappears, and each judge has a different failure signature. Llama-3 drifts to slot A even on byte-identical pairs. Mistral is clean on identical pairs but locks to a slot on 4 out of 4 paraphrase items. Qwen2 slot-locks on 2 out of 4 identical pairs.
3. Llama-3 and Mistral end up with the same aggregate score (0.333) while failing in different conditions, which seems to support the datasheet idea of reporting per-condition breakdowns rather than a single number.
4. A budget-side observation: Llama-3 declares a chat eos token that mlx-lm's default stop set misses, so before aligning it every call burned the full 512-token budget with identical verdicts (2.8x tokens, 2.6x latency). Each JSONL record now carries token counts, tps, peak memory and finish_reason.

## Caveats

- 12 items is a smoke-scale slice, meant to validate the harness and the per-condition breakdown, not to rank models.
- The slice is self-curated. I will gladly swap it for any benchmark slice or output schema you prefer.
- Independent pilot, no proprietary API calls anywhere in the pipeline.

All per-call records and summaries are committed in the repo under `results/`.

Best,
Orazio
