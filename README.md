# Dark Current MLX Evaluator

A small, local, **reproducible** harness for measuring **position-swap inconsistency** in
LLM-as-a-judge models — the bias where a judge's verdict changes just because two answers
swap places. It runs offline on Apple Silicon (Apple MLX), writes one JSONL record per judge
call, and reports a single `dark_current_score`.

> Status: **first real runs complete**. Three open-weight judges evaluated locally on a
> 12-item mini-slice (see Results); per-call records committed under `results/`. A
> paper-aligned benchmark slice is the next milestone. This is independent work, not an
> official reproduction (see *Scope*).

## Why this exists

Evaluating open-source judges usually drifts toward paid API baselines or GPU-heavy
pipelines. This harness keeps the first pass **local, deterministic, and auditable** on a
laptop:

- run each pairwise item twice — original order (A/B) and swapped order (B/A);
- parse strict `[[A]]` / `[[B]]` / `[[Tie]]` verdicts from the judge's free text;
- treat a judge as *consistent* only if it flips `A → B`, flips `B → A`, and keeps `Tie → Tie`;
- record hardware, prompt template, seed, and generation config alongside every call;
- aggregate a strict `dark_current_score` = fraction of items where the judge was *not*
  order-invariant (strict: an item with an unparseable verdict counts too, so the Results
  below report order-dependent and unparseable items separately).

## Results

### Real judge runs — mini-slice v0 (2026-07-03)

Three open-weight judges, fully local (MacBook M3 Pro 18 GB, `mlx-lm` 0.31.3), greedy decoding
(`temperature=0.0`, `seed=42`, `max_tokens=512`). The mini-slice
([`data/minislice_v0.jsonl`](data/minislice_v0.jsonl)) has 12 hand-written pairwise items in
three conditions: `clear_gap` (one answer plainly better, better side balanced 2/2),
`identical` (byte-identical answers — any A/B pick is pure position signal), `paraphrase`
(same content, different wording). 12 items × 2 orders = 24 judge calls per model.

| Judge (4-bit MLX) | clear_gap | identical | paraphrase | **order-dependent** | unparseable | strict score | gen tokens | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Meta-Llama-3-8B-Instruct | 0/4 | 1/4 | 2/4 | **3/12 (0.250)** | 1 | 0.333 | 4,459 | 170 s |
| Mistral-7B-Instruct-v0.3 | 0/4 | 0/4 | 3/4 | **3/12 (0.250)** | 1 | 0.333 | 2,725 | 106 s |
| Qwen2-7B-Instruct | 0/4 | 2/4 | 3/4 | **5/12 (0.417)** | 0 | 0.417 | 1,508 | 71 s |

Condition cells count *order-dependent* items: both verdicts parsed, and the verdict changed
with the A/B order. *Unparseable* items have one call without the strict marker, and neither
call was truncated: Llama-3 answered a `clear_gap` item correctly in prose but skipped `[[B]]`;
Mistral wrote `[Tie]` with single brackets on a paraphrase item it had called a tie in the other
order. The *strict score* is the `dark_current_score` the CLI summary reports, which counts
unparseable items as not order-invariant (it was this README's headline number until
2026-10-24).

What the aggregate score hides:

- **All three judges read a real quality signal well**: no order-dependent `clear_gap` item, and
  the parsed `clear_gap` verdicts picked the objectively better answer in both orders 11/12
  times (the twelfth is Llama-3's unparseable call).
- **Position bias appears exactly where the signal disappears, with a different signature per
  judge**: Llama-3 picks slot A in both orders on 1/4 identical and 2/4 paraphrase items;
  Mistral is order-invariant until answers are merely *equivalent*, then order-dependent on 3/4
  paraphrase items; Qwen2 picks slot A in both orders on 2/4 identical items and is
  order-dependent on 3/4 paraphrases.
- **Same aggregate, different pathology**: Llama-3 and Mistral share the same score (3/12
  order-dependent) while failing in different conditions — the case for per-condition
  breakdowns over a single number.

Reproduce the breakdown (order-invariant / order-dependent / unparseable per condition, plus
the strict score): `python3 scripts/category_breakdown.py results/*_minislice_v0.jsonl`.
Full per-call records (hardware metadata, generation config, usage stats, raw generations,
parsed verdicts) are committed under [`results/`](results/).

> **Budget note.** The runner aligns the tokenizer-declared chat eos token with the `mlx-lm`
> stop set (Llama-3 declares `<|eot_id|>`, whose id is missing from the wrapper's stop ids).
> Before the fix every call burned the full 512-token budget with identical verdicts: 12,288
> vs 4,459 generated tokens and 435 s vs 170 s. Per-call `usage` records token counts, tps,
> peak memory, and `finish_reason`; summaries aggregate them in `usage_totals`.

### Sanity baselines (mock judges)

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
  --input data/minislice_v0.jsonl \
  --output results/llama3_8b_4bit_minislice_v0.jsonl \
  --report results/llama3_8b_4bit_minislice_v0_summary.json
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
  "timing":  {"latency_s": 0.0},
  "usage":   {"prompt_tokens": 310, "generation_tokens": 142, "prompt_tps": 250.0, "generation_tps": 28.5, "peak_memory_gb": 5.1, "finish_reason": "stop"}
}
```

`parsed_position_winner` is which *slot* won (A/B/tie); `parsed_original_winner` maps that back
to the original model id, so a swap-inconsistent judge is visible at the item level.

## Repository structure

```
src/dark_current_mlx/
  cli.py        # argparse CLI (entry point: dark-current-mlx)
  pipeline.py   # A/B + B/A runs, JSONL writer, summary metrics + usage totals
  runners.py    # MockRunner (offline) + lazy MlxRunner (sampler API, eos alignment, usage stats)
  verdicts.py   # strict [[A]]/[[B]]/[[Tie]] parser + swap-consistency logic
  models.py     # PairwiseItem / Verdict / GenerationConfig dataclasses
  prompts.py    # pairwise_judge_v1 template + A/B-swap renderer
  dataset.py    # JSONL loader (+ legacy schema) and builtin sample
eval_pipeline.py        # backward-compatible entry point (no install needed)
data/sample_pairwise.jsonl   # 2-item dry-run sample
data/minislice_v0.jsonl      # 12-item mini-slice (clear_gap / identical / paraphrase)
scripts/category_breakdown.py  # per-condition order-dependent / unparseable split from run JSONL
schema.json             # output record shape
tests/                  # 8 offline unit tests (no network / no MLX / no downloads)
docs/                   # results-note template + drafts
results/                # committed: the three mini-slice v0 runs (records + summaries)
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
| Real MLX judge run on a mini-slice (3 open judges, fixed seed) | ✅ done (see Results) |
| Budget-aware usage fields (tokens, tps, peak memory, finish reason) | ✅ done (scaffold/routing dimensions planned) |
| Order-dependent vs unparseable split | ✅ in `scripts/category_breakdown.py`; 🔜 in the CLI summary |
| Compact results note for the paper authors | ✅ sent 2026-07-03 (draft in `docs/`) |
| Paper-aligned benchmark adapter (MT-Bench / AlpacaEval-style) | 🔜 planned |

## Scope & attribution

Inspired by *"LLM Judges Have Dark Current: A Psychometric Datasheet for LLM-as-a-Judge
Evaluation"* (Usami et al.). This is an **independent pilot** exploring local reproducibility
on Apple Silicon. It is **not endorsed by the authors** and does **not** reproduce the paper's
full results. The authors acknowledge it in the README of their revised repository,
[`dark-current-v2-review`](https://github.com/usamihiroyasu/dark-current-v2-review). No
proprietary API calls are used anywhere in the pipeline.

## License

MIT — see [`LICENSE`](LICENSE).
