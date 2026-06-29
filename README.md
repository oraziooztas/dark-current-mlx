# Dark Current MLX Evaluator

A small, local, **reproducible** harness for measuring **position-swap inconsistency** in
LLM-as-a-judge models — the bias where a judge's verdict changes just because two answers
swap places. It runs offline on Apple Silicon (Apple MLX), writes one JSONL record per judge
call, and reports a single `dark_current_score`.

> Status: **pilot scaffold**. The plumbing, metric, and tests are working and verified
> (see Results). Real open-source judge runs and a paper-aligned benchmark slice are the
> next milestone. This is independent work, not an official reproduction (see *Scope*).

## Why this exists

Evaluating open-source judges usually drifts toward paid API baselines or GPU-heavy
pipelines. This harness keeps the first pass **local, deterministic, and auditable** on a
laptop:

- run each pairwise item twice — original order (A/B) and swapped order (B/A);
- parse strict `[[A]]` / `[[B]]` / `[[Tie]]` verdicts from the judge's free text;
- treat a judge as *consistent* only if it flips `A → B`, flips `B → A`, and keeps `Tie → Tie`;
- record hardware, prompt template, seed, and generation config alongside every call;
- aggregate a strict `dark_current_score` = fraction of items where the judge was *not*
  order-invariant.

## Results (verified mock baselines)

Two deterministic mock judges validate the metric end-to-end — no model download, no network.
A purely position-biased judge should score `1.0` (maximally inconsistent); an order-invariant
judge should score `0.0`. Both hold:

| Mock judge baseline                | Items | Swap-consistent | `dark_current_score` |
|------------------------------------|:-----:|:---------------:|:--------------------:|
| `position` (always picks slot A)   |   2   |        0        |       **1.00**       |
| `tie` (always `[[Tie]]`)           |   2   |        2        |       **0.00**       |

```
$ python3 -m unittest discover -s tests   →  Ran 8 tests ... OK  (~0.02s)
```

These mocks are *sanity baselines*, not real judges: they prove the pipeline, parser, and
metric behave correctly before any model is loaded. Plugging in a real MLX judge is a
one-flag change (`--runner mlx --model ...`).

## Quickstart

### 1. Offline dry run (no MLX, no model)

Validates dataset loading, prompt rendering, JSONL output, and the metric:

```bash
python3 eval_pipeline.py --runner mock --input data/sample_pairwise.jsonl
# writes results/run.jsonl and results/summary.json, prints the summary
```

### 2. Run the tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

### 3. Real MLX judge run

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[mlx]"

python3 eval_pipeline.py \
  --runner mlx \
  --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
  --input data/sample_pairwise.jsonl \
  --output results/llama3_8b_4bit.jsonl \
  --report results/llama3_8b_4bit_summary.json
```

## Dataset format

Input is JSONL, one pairwise item per line:

```json
{"item_id": "sample_fibonacci", "instruction": "Write a Python function for Fibonacci numbers.", "response_a": "...", "response_b": "...", "model_a_id": "toy_recursive", "model_b_id": "toy_iterative", "metadata": {"source": "local_sample"}}
```

Legacy prototype field names (`id`, `model_a_response`, `model_b_response`) are also accepted,
so older data files still load. Use `--input builtin` for an in-memory two-item smoke set.

## Output schema

One JSONL record per judge call (`order` is `ab` or `ba`); see [`schema.json`](schema.json):

```json
{
  "eval_id": "uuid shared across the run",
  "item_id": "sample_fibonacci",
  "judge_model": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
  "hardware_env": {"platform": "macOS-...-arm64", "machine": "arm64", "python_version": "3.x"},
  "generation_config": {"max_tokens": 256, "temperature": 0.0, "seed": 42},
  "prompt_template_id": "pairwise_judge_v1",
  "order": "ab",
  "inputs":  {"model_a_id": "...", "model_b_id": "...", "instruction": "...", "response_a": "...", "response_b": "...", "metadata": {}},
  "outputs": {"raw_generation": "... [[A]]", "parsed_position_winner": "A", "parsed_original_winner": "model_a"},
  "timing":  {"latency_s": 0.0}
}
```

`parsed_position_winner` is which *slot* won (A/B/tie); `parsed_original_winner` maps that back
to the original model id, so a swap-inconsistent judge is visible at the item level.

## Repository structure

```
src/dark_current_mlx/
  cli.py        # argparse CLI (entry point: dark-current-mlx)
  pipeline.py   # A/B + B/A runs, JSONL writer, summary metrics
  runners.py    # MockRunner (offline) + lazy MlxRunner (no import until used)
  verdicts.py   # strict [[A]]/[[B]]/[[Tie]] parser + swap-consistency logic
  models.py     # PairwiseItem / Verdict / GenerationConfig dataclasses
  prompts.py    # pairwise_judge_v1 template + A/B-swap renderer
  dataset.py    # JSONL loader (+ legacy schema) and builtin sample
eval_pipeline.py        # backward-compatible entry point (no install needed)
data/sample_pairwise.jsonl
schema.json             # output record shape
tests/                  # 8 offline unit tests (no network / no MLX / no downloads)
docs/                   # results-note template
```

The `MlxRunner` imports `mlx-lm` lazily, so the tests and the mock runner never touch MLX —
that is why the whole suite runs in well under a second with zero downloads.

## Status & roadmap

| Area | State |
|------|-------|
| Package + CLI (`mock` / `mlx` runners) | ✅ done |
| Strict verdict parser + swap-consistency metric | ✅ done |
| JSONL output + reproducibility metadata + summary | ✅ done |
| Offline test suite (8 tests) | ✅ done |
| Real MLX judge run on a mini-slice (2–3 open judges, fixed seed) | 🔜 next |
| Paper-aligned benchmark adapter (MT-Bench / AlpacaEval-style) | 🔜 planned |
| Budget-aware fields (`inference_budget_tokens`, `wall_clock_s`, scaffold/routing) | 🔜 planned |
| Compact results note for the paper authors | 🔜 planned |

## Scope & attribution

Inspired by *"LLM Judges Have Dark Current: A Psychometric Datasheet for LLM-as-a-Judge
Evaluation"* (Usami et al.). This is an **independent pilot** exploring local reproducibility
on Apple Silicon. It is **not affiliated with or endorsed by the authors** and does **not**
reproduce the paper's full results. No proprietary API calls are used anywhere in the pipeline.

## License

MIT — see [`LICENSE`](LICENSE).
